# Reference Teacher Labeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Safely run the Git-external YOLOv8s reference checkpoint on the 100 selected scenes and produce an auditable, non-truth-mutating review pack.

**Architecture:** A pure `reference_teacher` module validates and normalizes external predictions without importing Ultralytics. A research-only runner lazily imports PyTorch and Ultralytics, restricts checkpoint globals, uses `weights_only=True`, and writes raw predictions. A separate pack builder renders Git-external COCO proposals, overlays and contact sheets while proving the formal truth file hash is unchanged.

**Tech Stack:** Python 3.12 core, pytest, Pillow/OpenCV, JSON/COCO; isolated Python 3.11 + PyTorch 2.7.1 + Ultralytics 8.2.40 for reference inference.

---

### Task 1: Teacher prediction contract

**Files:**
- Create: `ml/src/crrc_vision/reference_teacher.py`
- Create: `ml/tests/test_reference_teacher.py`

- [ ] **Step 1: Write failing contract tests**

```python
from crrc_vision.reference_teacher import (
    TeacherPrediction,
    ensure_complete_selection,
    map_teacher_category,
    validate_checkpoint_globals,
)


def test_checkpoint_globals_reject_non_framework_types():
    assert validate_checkpoint_globals([
        "torch.nn.modules.conv.Conv2d",
        "ultralytics.nn.tasks.DetectionModel",
    ]) == ()
    assert validate_checkpoint_globals(["os.system"]) == ("UNSAFE_CHECKPOINT_GLOBAL",)


def test_teacher_mapping_is_explicitly_unconfirmed():
    category, status = map_teacher_category(1)
    assert category == "pipe_joint"
    assert status == "inferred_unconfirmed"


def test_prediction_id_is_stable_and_preserves_teacher_class():
    item = TeacherPrediction("a.jpg", 2, "class_2", (1, 2, 3, 4), 0.9)
    assert item.stable_id == item.stable_id
    assert item.to_dict()["teacher_class_id"] == 2


def test_selection_coverage_rejects_missing_and_extra_images():
    assert ensure_complete_selection(["a.jpg", "b.jpg"], ["a.jpg", "b.jpg"]) == ()
    assert ensure_complete_selection(["a.jpg", "b.jpg"], ["a.jpg"]) == (
        "INCOMPLETE_SELECTION_COVERAGE",
    )
```

- [ ] **Step 2: Run RED**

Run: `.\.venv\Scripts\python.exe -m pytest ml\tests\test_reference_teacher.py -v`

Expected: FAIL with `ModuleNotFoundError: crrc_vision.reference_teacher`.

- [ ] **Step 3: Implement the pure contract**

```python
import hashlib
from dataclasses import asdict, dataclass

ALLOWED_GLOBAL_PREFIXES = ("torch.nn.modules.", "ultralytics.")
TEACHER_CATEGORY_MAP = {0: "fastener", 1: "pipe_joint", 2: "fastener"}


def validate_checkpoint_globals(names):
    return () if all(str(name).startswith(ALLOWED_GLOBAL_PREFIXES) for name in names) else (
        "UNSAFE_CHECKPOINT_GLOBAL",
    )


def map_teacher_category(class_id):
    if class_id not in TEACHER_CATEGORY_MAP:
        raise ValueError(f"unsupported teacher class: {class_id}")
    return TEACHER_CATEGORY_MAP[class_id], "inferred_unconfirmed"


def ensure_complete_selection(expected, actual):
    return () if sorted(expected) == sorted(actual) else ("INCOMPLETE_SELECTION_COVERAGE",)


@dataclass(frozen=True)
class TeacherPrediction:
    relative_path: str
    teacher_class_id: int
    teacher_class_name: str
    bbox: tuple[float, float, float, float]
    score: float

    @property
    def stable_id(self):
        raw = f"{self.relative_path}|{self.teacher_class_id}|{self.bbox}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def to_dict(self):
        category, mapping_status = map_teacher_category(self.teacher_class_id)
        return {
            "id": self.stable_id,
            **asdict(self),
            "bbox": list(self.bbox),
            "mapped_category": category,
            "mapping_status": mapping_status,
            "review_status": "unreviewed",
        }
```

- [ ] **Step 4: Run GREEN and full Python tests**

Run: `.\.venv\Scripts\python.exe -m pytest ml\tests\test_reference_teacher.py -v`

Expected: 4 tests pass.

Run: `.\.venv\Scripts\python.exe -m pytest ml\tests -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add ml/src/crrc_vision/reference_teacher.py ml/tests/test_reference_teacher.py
git commit -m "feat: add reference teacher prediction contract"
```

### Task 2: Restricted checkpoint runner

**Files:**
- Create: `ml/scripts/run_reference_teacher.py`
- Modify: `ml/tests/test_reference_teacher.py`

- [ ] **Step 1: Write failing manifest tests**

```python
from crrc_vision.reference_teacher import build_run_manifest


def test_run_manifest_records_integrity_and_research_boundary():
    value = build_run_manifest(
        checkpoint_sha256="A" * 64,
        selection_sha256="B" * 64,
        truth_sha256="C" * 64,
        images=100,
        predictions=700,
    )
    assert value["safe_load"] == "weights_only"
    assert value["research_only"] is True
    assert value["truth_sha256_before"] == value["truth_sha256_after"]
```

- [ ] **Step 2: Run RED**

Run: `.\.venv\Scripts\python.exe -m pytest ml\tests\test_reference_teacher.py::test_run_manifest_records_integrity_and_research_boundary -v`

Expected: FAIL because `build_run_manifest` is missing.

- [ ] **Step 3: Add the manifest helper and thin runner**

`build_run_manifest()` returns the supplied hashes, `safe_load="weights_only"`,
`mapping_status="inferred_unconfirmed"`, `research_only=True`, image/prediction counts and identical
before/after truth hashes.

The CLI accepts explicit `--checkpoint`, `--runtime-version`, `--selection`, `--truth`, `--source`,
`--output`, `--imgsz`, `--conf` and `--device`. It must:

1. resolve every output below `CRRC_VISION_DATA_ROOT`;
2. require the installed Ultralytics version to equal `8.2.40`;
3. call `torch.serialization.get_unsafe_globals_in_checkpoint()` and reject any name that fails
   `validate_checkpoint_globals()`;
4. import only the enumerated allowed framework classes and enter `torch.serialization.safe_globals()`;
5. call `torch.load(..., map_location="cpu", weights_only=True)`;
6. run all 100 selected images and write xywh predictions plus per-image timing;
7. write atomically to `raw-predictions.json` and `run-manifest.json`;
8. hash the truth file before and after and exit nonzero if it changed.

- [ ] **Step 4: Verify tests and CLI help**

Run: `.\.venv\Scripts\python.exe -m pytest ml\tests -q`

Expected: all tests pass.

Run: `.\.venv\Scripts\python.exe ml\scripts\run_reference_teacher.py --help`

Expected: exit 0 without importing Ultralytics.

- [ ] **Step 5: Commit**

```powershell
git add ml/src/crrc_vision/reference_teacher.py ml/tests/test_reference_teacher.py ml/scripts/run_reference_teacher.py
git commit -m "feat: add restricted reference teacher runner"
```

### Task 3: Review-pack builder

**Files:**
- Create: `ml/scripts/build_reference_teacher_pack.py`
- Modify: `ml/src/crrc_vision/reference_teacher.py`
- Modify: `ml/tests/test_reference_teacher.py`

- [ ] **Step 1: Write failing COCO conversion tests**

```python
from crrc_vision.reference_teacher import build_proposal_document


def test_proposal_document_never_marks_teacher_boxes_as_truth():
    selection = [{"relative_path": "a.jpg", "scene_group": "g1", "split": "train"}]
    manifest = {"a.jpg": {"width": 100, "height": 80}}
    predictions = [{
        "id": "p1", "relative_path": "a.jpg", "teacher_class_id": 2,
        "teacher_class_name": "class_2", "bbox": [1, 2, 3, 4], "score": 0.9,
        "mapped_category": "fastener", "mapping_status": "inferred_unconfirmed",
        "review_status": "unreviewed",
    }]
    value = build_proposal_document(selection, manifest, predictions)
    assert value["images"][0]["image_review_status"] == "unreviewed"
    assert value["annotations"][0]["review_status"] == "unreviewed"
    assert value["annotations"][0]["proposal_source"] == "reference-yolov8s"
```

- [ ] **Step 2: Run RED**

Run: `.\.venv\Scripts\python.exe -m pytest ml\tests\test_reference_teacher.py::test_proposal_document_never_marks_teacher_boxes_as_truth -v`

Expected: FAIL because `build_proposal_document` is missing.

- [ ] **Step 3: Implement conversion and renderer**

`build_proposal_document()` assigns deterministic image/annotation IDs, fixed categories
`fastener/pipe_joint`, preserves teacher metadata and clips every xywh box to image bounds.

The CLI reads the raw predictions and writes below
`review-packs/fastener-v2/reference-teacher-v1/`:

- `teacher-proposals.json`;
- `instances.proposals.json`;
- `review-index.csv`;
- `overlays/<image>.jpg` with numbered boxes;
- `candidate-sheets/candidates-NN.jpg` with candidate ID, class and score;
- `contact-sheets/full-images-NN.jpg` with full-image context;
- `pack-manifest.json` with output hashes and counts.

It refuses incomplete selection coverage and proves the formal truth file hash is unchanged.

- [ ] **Step 4: Verify tests and CLI help**

Run: `.\.venv\Scripts\python.exe -m pytest ml\tests -q`

Expected: all tests pass.

Run: `.\.venv\Scripts\python.exe ml\scripts\build_reference_teacher_pack.py --help`

Expected: exit 0.

- [ ] **Step 5: Commit**

```powershell
git add ml/src/crrc_vision/reference_teacher.py ml/tests/test_reference_teacher.py ml/scripts/build_reference_teacher_pack.py
git commit -m "feat: build reference teacher review pack"
```

### Task 4: Execute the 100-image private run

**Files:**
- Git-external output: `review-packs/fastener-v2/reference-teacher-v1/`
- Git-external output: `runs/reference-teacher-v1/`

- [ ] **Step 1: Record the current truth hash**

Run: `Get-FileHash -Algorithm SHA256 "$env:CRRC_VISION_DATA_ROOT\annotations\fastener-v2\instances.json"`

Expected: `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`.

- [ ] **Step 2: Run safe reference inference**

Run the isolated Python 3.11 runtime with `PYTHONPATH` pointing at `ml/src`, checkpoint
`references/Bolt_Detection_2026-08-25/dtetct_best.pt`, selection `selections/selection-v2.json`, source
`source/20240529-luosi`, output `runs/reference-teacher-v1`, input 640, confidence 0.25 and CUDA device 0.

Expected: exactly 100 processed images and a raw prediction JSON; truth hash unchanged.

- [ ] **Step 3: Build private review assets**

Run: `.\.venv\Scripts\python.exe ml\scripts\build_reference_teacher_pack.py --predictions runs/reference-teacher-v1/raw-predictions.json`

Expected: 100 overlays, full-image contact sheets, candidate sheets, complete selection coverage and no formal truth mutation.

- [ ] **Step 4: Audit candidate and full-image sheets**

Inspect every candidate sheet for obvious false positives and every full-image sheet for obvious misses. Write
`ai-review-v1.json` with candidate decisions and image-level `needs_manual` until whole-image completeness is proven.
Do not change `annotations/fastener-v2/instances.json` in this task.

### Task 5: Evidence and handoff

**Files:**
- Modify: `docs/validation/2026-08-25-full-image-v2-bootstrap.md`
- Modify: `PROJECT_STATUS.md`

- [ ] **Step 1: Run fresh verification**

```powershell
.\.venv\Scripts\python.exe -m pytest ml\tests -q
git diff --check
git status --short
```

Expected: all tests pass; only intended docs are uncommitted; private data is absent from Git status.

- [ ] **Step 2: Record evidence**

Record checkpoint/selection/truth/output hashes, image and proposal counts, candidate review counts, images needing
manual completion, runtime timing and the unchanged training-gate result. Never report candidate count as recall.

- [ ] **Step 3: Update status and commit**

```powershell
git add docs/validation/2026-08-25-full-image-v2-bootstrap.md PROJECT_STATUS.md
git commit -m "docs: record reference teacher labeling evidence"
```
