# Full-image Fastener Vision V2 Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a deterministic 100-scene physical-fastener labeling pack, a truth-quality gate, and a guarded 12-image open-vocabulary proposal pilot.

**Architecture:** Core Python stays model-independent and fully tested. Private manifests, COCO files, overlays and review decisions remain below `CRRC_VISION_DATA_ROOT`; the optional GroundingDINO dependency is imported only by its CLI. PicoDet training and Android inference are separate follow-on plans because trustworthy labels are their prerequisite.

**Tech Stack:** Python 3.12, pytest, OpenCV, COCO JSON, optional Hugging Face Transformers GroundingDINO.

---

### Task 1: Deterministic representative-frame selection

**Files:**
- Create: `ml/src/crrc_vision/selection.py`
- Create: `ml/tests/test_selection.py`
- Create: `ml/scripts/build_fastener_selection.py`

- [ ] **Step 1: Write failing tests**

```python
from crrc_vision.selection import SelectionCandidate, select_representatives


def row(path, group, split, focus, count):
    return SelectionCandidate(path, group, split, focus, count)


def test_selection_is_deterministic_and_unique_by_group():
    rows = [
        row("a1.jpg", "g1", "train", 10.0, 1),
        row("a2.jpg", "g1", "train", 20.0, 1),
        row("b.jpg", "g2", "train", 15.0, 8),
        row("c.jpg", "g3", "val", 12.0, 3),
        row("d.jpg", "g4", "val", 11.0, 0),
    ]
    first = select_representatives(rows, target=4, val_count=2)
    assert first == select_representatives(list(reversed(rows)), target=4, val_count=2)
    assert len({item.scene_group for item in first}) == 4
    assert "a2.jpg" in {item.relative_path for item in first}
    assert sum(item.split == "val" for item in first) == 2


def test_selection_rejects_impossible_target():
    try:
        select_representatives([row("a.jpg", "g1", "train", 1.0, 0)], target=2, val_count=1)
    except ValueError as error:
        assert "target" in str(error)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run RED**

Run: `\.venv\Scripts\python.exe -m pytest ml/tests/test_selection.py -v`

Expected: `ModuleNotFoundError: crrc_vision.selection`.

- [ ] **Step 3: Implement minimal selector**

```python
from dataclasses import asdict, dataclass


@dataclass(frozen=True, order=True)
class SelectionCandidate:
    relative_path: str
    scene_group: str
    split: str
    focus_score: float
    candidate_count: int

    def to_dict(self):
        return asdict(self)


def select_representatives(rows, *, target=100, val_count=20):
    best = {}
    for item in sorted(rows, key=lambda value: value.relative_path):
        old = best.get(item.scene_group)
        if old is None or (item.focus_score, -abs(item.candidate_count - 3), item.relative_path) > (
            old.focus_score, -abs(old.candidate_count - 3), old.relative_path
        ):
            best[item.scene_group] = item
    values = sorted(best.values(), key=lambda value: value.scene_group)
    if target < 1 or target > len(values):
        raise ValueError("target must fit available scene groups")
    val = [item for item in values if item.split == "val"]
    train = [item for item in values if item.split == "train"]
    if val_count > len(val) or target - val_count > len(train):
        raise ValueError("target split quota is unavailable")
    rank = lambda item: (-item.candidate_count, -item.focus_score, item.scene_group)
    return sorted(sorted(val, key=rank)[:val_count] + sorted(train, key=rank)[:target-val_count])
```

- [ ] **Step 4: Add CLI and generate selection**

The CLI reads `manifest.jsonl` and COCO candidate counts, calls the tested function, and atomically writes:

```python
payload = {"version": "selection-v2", "target": args.target,
           "val_count": args.val_count, "items": [item.to_dict() for item in selected]}
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
temporary.replace(output)
```

Run: `\.venv\Scripts\python.exe ml/scripts/build_fastener_selection.py`

Expected: 100 unique groups, 80 train and 20 val.

- [ ] **Step 5: Verify and commit**

Run: `\.venv\Scripts\python.exe -m pytest ml/tests -v`

Commit: `git commit -m "feat: select fastener labeling representatives"`

### Task 2: Physical-fastener truth contract and label pack

**Files:**
- Create: `ml/src/crrc_vision/fastener_annotations.py`
- Create: `ml/tests/test_fastener_annotations.py`
- Create: `ml/scripts/build_fastener_label_pack.py`

- [ ] **Step 1: Write failing gate tests**

```python
from crrc_vision.fastener_annotations import evaluate_fastener_truth


def sample(status="accept"):
    return {"images": [
        {"id": 1, "file_name": "a.jpg", "scene_group": "g1", "split": "train", "image_review_status": status},
        {"id": 2, "file_name": "b.jpg", "scene_group": "g2", "split": "val", "image_review_status": status}],
        "categories": [{"id": 1, "name": "fastener"}, {"id": 2, "name": "pipe_joint"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [1, 2, 3, 4], "review_status": status}]}


def test_gate_rejects_unreviewed_images():
    assert "UNREVIEWED_IMAGE" in evaluate_fastener_truth(sample("unreviewed"), minimum_groups=2).error_codes


def test_gate_accepts_reviewed_train_and_val():
    report = evaluate_fastener_truth(sample(), minimum_groups=2)
    assert report.can_train is True
    assert report.accepted_boxes == 1


def test_gate_rejects_scene_leakage():
    value = sample()
    value["images"][1]["scene_group"] = "g1"
    assert "SCENE_GROUP_LEAKAGE" in evaluate_fastener_truth(value, minimum_groups=1).error_codes
```

- [ ] **Step 2: Run RED**

Run: `\.venv\Scripts\python.exe -m pytest ml/tests/test_fastener_annotations.py -v`

Expected: missing-module failure.

- [ ] **Step 3: Implement validator**

```python
from dataclasses import asdict, dataclass

VALID_CATEGORIES = {1: "fastener", 2: "pipe_joint"}
VALID_REVIEW = {"accept", "accept_empty", "reject", "needs_manual", "unreviewed"}


@dataclass(frozen=True)
class FastenerTruthReport:
    reviewed_groups: int
    accepted_boxes: int
    error_codes: tuple[str, ...]

    @property
    def can_train(self):
        return not self.error_codes

    def to_dict(self):
        return {**asdict(self), "can_train": self.can_train}


def evaluate_fastener_truth(document, *, minimum_groups=80):
    errors = set()
    images, annotations = document.get("images", []), document.get("annotations", [])
    if {row.get("id"): row.get("name") for row in document.get("categories", [])} != VALID_CATEGORIES:
        errors.add("INVALID_CATEGORIES")
    if any(row.get("image_review_status") == "unreviewed" for row in images):
        errors.add("UNREVIEWED_IMAGE")
    if any(row.get("image_review_status") not in VALID_REVIEW for row in images):
        errors.add("INVALID_IMAGE_REVIEW")
    accepted = [row for row in annotations if row.get("review_status") == "accept"]
    if not accepted:
        errors.add("NO_ACCEPTED_BOX")
    groups = {row.get("scene_group") for row in images}
    if len(groups) < minimum_groups:
        errors.add("INSUFFICIENT_REVIEWED_GROUPS")
    splits = {}
    for row in images:
        splits.setdefault(row["scene_group"], set()).add(row["split"])
    if any(len(value) > 1 for value in splits.values()):
        errors.add("SCENE_GROUP_LEAKAGE")
    if not any(row.get("split") == "val" for row in images):
        errors.add("EMPTY_VALIDATION")
    return FastenerTruthReport(len(groups), len(accepted), tuple(sorted(errors)))
```

- [ ] **Step 4: Add label-pack CLI**

Generate COCO images with `image_review_status="unreviewed"`, fixed categories, no annotations, plus `review-index.csv` and Label Studio tasks. Resolve output below `asset_root()` and reject escaping paths.

Run: `\.venv\Scripts\python.exe ml/scripts/build_fastener_label_pack.py`

Expected: Git-external `annotations/fastener-v2/instances.json` and `review-packs/fastener-v2/label-studio-tasks.json`.

- [ ] **Step 5: Verify and commit**

Run: `\.venv\Scripts\python.exe -m pytest ml/tests -v`

Commit: `git commit -m "feat: add physical fastener truth contract"`

### Task 3: Guarded proposal pilot

**Files:**
- Create: `ml/src/crrc_vision/proposals.py`
- Create: `ml/tests/test_proposals.py`
- Create: `ml/scripts/run_grounding_dino_pilot.py`

- [ ] **Step 1: Write failing primitive tests**

```python
from crrc_vision.proposals import PilotAudit, Proposal, clip_box, pilot_can_expand


def test_clip_box_stays_inside_image():
    assert clip_box((-2.0, 3.0, 15.0, 12.0), width=10, height=10) == (0.0, 3.0, 10.0, 7.0)


def test_pilot_requires_precision_and_coverage():
    assert pilot_can_expand(PilotAudit(7, 5, 3, 12)) is True
    assert pilot_can_expand(PilotAudit(5, 7, 3, 12)) is False
    assert pilot_can_expand(PilotAudit(8, 2, 4, 12)) is False


def test_proposal_id_is_stable():
    item = Proposal("a.jpg", "fastener", (1.0, 2.0, 3.0, 4.0), 0.8, "grounding-dino")
    assert item.stable_id == item.stable_id
```

- [ ] **Step 2: Run RED**

Run: `\.venv\Scripts\python.exe -m pytest ml/tests/test_proposals.py -v`

Expected: missing-module failure.

- [ ] **Step 3: Implement primitives**

```python
import hashlib
from dataclasses import dataclass


def clip_box(box, *, width, height):
    x, y, w, h = box
    left, top = max(0.0, x), max(0.0, y)
    right, bottom = min(float(width), x + w), min(float(height), y + h)
    return left, top, max(0.0, right-left), max(0.0, bottom-top)


@dataclass(frozen=True)
class Proposal:
    relative_path: str
    category: str
    bbox: tuple[float, float, float, float]
    score: float
    source: str

    @property
    def stable_id(self):
        raw = f"{self.relative_path}|{self.category}|{self.bbox}|{self.source}".encode()
        return hashlib.sha256(raw).hexdigest()[:16]


@dataclass(frozen=True)
class PilotAudit:
    accepted: int
    rejected: int
    images_with_missed_targets: int
    images: int


def pilot_can_expand(audit):
    reviewed = audit.accepted + audit.rejected
    precision = audit.accepted / reviewed if reviewed else 0.0
    missed = audit.images_with_missed_targets / audit.images if audit.images else 1.0
    return precision >= 0.50 and missed <= 0.30
```

- [ ] **Step 4: Add optional model CLI**

Import `torch` and `transformers` inside `main()`, use model `IDEA-Research/grounding-dino-tiny`, prompt `bolt. nut. screw. fastener. pipe joint.`, and 12 images stratified by split/candidate count. Write JSON and overlays under `review-packs/fastener-v2/pilot`. `--expand` must read `pilot-audit.json` and exit 2 unless the tested gate passes.

Run: `\.venv\Scripts\python.exe ml/scripts/run_grounding_dino_pilot.py`

Expected: 12-image proposal pack, or stable `MODEL_DEPENDENCY_UNAVAILABLE` without truth mutation.

- [ ] **Step 5: Audit every proposal and commit**

Write `pilot-audit.json` with accepted, rejected, missed-target images, decision and reviewer. FAIL preserves the manual pack and cannot process the other 88 images.

Run: `\.venv\Scripts\python.exe -m pytest ml/tests -v`

Commit: `git commit -m "feat: add guarded fastener proposal pilot"`

### Task 4: Evidence and handoff

**Files:**
- Create: `docs/validation/2026-08-25-full-image-v2-bootstrap.md`
- Modify: `PROJECT_STATUS.md`
- Modify: `README.md`

- [ ] **Step 1: Run fresh verification**

```powershell
$env:CRRC_VISION_DATA_ROOT='E:\Work\京新数智\识动hicool\中车眼镜数据资产'
.\.venv\Scripts\python.exe -m pytest ml\tests -v
.\.venv\Scripts\python.exe ml\scripts\build_fastener_selection.py
.\.venv\Scripts\python.exe ml\scripts\build_fastener_label_pack.py
git diff --check
```

Expected: tests pass; selection has 100 unique groups and 80/20 split; label pack is outside Git.

- [ ] **Step 2: Record exact evidence**

Record test count, file hashes, pilot gate result, and remaining empirical blockers. If unaudited, state `pilot gate pending`; never call proposal count accuracy.

- [ ] **Step 3: Update state and commit**

Set next action to exactly one evidence branch: `audit pilot`, `manual box labeling`, or `prepare PicoDet training plan`.

Commit: `git commit -m "docs: record full-image vision v2 bootstrap evidence"`
