# Safe Automatic Labeling Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a deterministic multi-source candidate, Codex arbitration, and AI-silver truth pipeline that processes all 482 field images without mutating formal truth.

**Architecture:** Normalize teacher, HSV, temporal, and later student detections into one candidate contract; fuse only genuinely independent source families; render two-pass Codex review assets; export only whole-image-complete decisions to a separate silver COCO file. Phase B training is deliberately gated on at least 80 complete scene groups (64 train and 16 val).

**Tech Stack:** Python 3.12, dataclasses, JSON/CSV, NumPy, OpenCV, Pillow, pytest; Git-external image/model assets through `CRRC_VISION_DATA_ROOT`.

---

## File map

- `ml/src/crrc_vision/auto_labeling.py`: normalized candidates, source-family rules, IoU clustering, weighted fusion, stable IDs.
- `ml/src/crrc_vision/temporal.py`: homography quality contract and same-scene box propagation.
- `ml/src/crrc_vision/codex_review.py`: two-pass review task and decision validation.
- `ml/src/crrc_vision/codex_review_pack.py`: deterministic grid, overlay, crop, and neighbor rendering.
- `ml/src/crrc_vision/silver_truth.py`: whole-image completeness gate and isolated COCO export.
- `ml/src/crrc_vision/tiles.py`: deterministic 2×2 overlap tiles and inverse coordinate mapping.
- `ml/scripts/build_safe_auto_candidates.py`: combine full/multiscale/tile teacher JSON, HSV anchors, and temporal propagation.
- `ml/scripts/build_codex_review_pack.py`: render full-image grids, overlays, crops, and review task JSON.
- `ml/scripts/apply_codex_review.py`: merge first/second-pass decisions and reject unsafe output.
- `ml/scripts/export_silver_truth.py`: evaluate the 80-scene gate and emit silver truth or a refusal report.
- `ml/tests/test_auto_labeling.py`, `test_temporal.py`, `test_codex_review.py`, `test_silver_truth.py`, `test_tiles.py`: focused contracts.

### Task 1: Normalized candidates and independent-source fusion

**Files:**
- Create: `ml/src/crrc_vision/auto_labeling.py`
- Create: `ml/tests/test_auto_labeling.py`

- [ ] **Step 1: Write failing fusion tests**

```python
from crrc_vision.auto_labeling import Candidate, fuse_candidates


def candidate(source_id: str, family: str, category: str, box: tuple[float, ...], score: float):
    return Candidate("a.jpg", source_id, family, category, box, score)


def test_multiscale_teacher_hits_are_one_source_family():
    fused = fuse_candidates([
        candidate("teacher-640", "reference_teacher", "fastener", (10, 10, 30, 30), 0.8),
        candidate("teacher-1280", "reference_teacher", "fastener", (11, 11, 31, 31), 0.9),
    ])
    assert fused[0].supporting_families == ("reference_teacher",)
    assert fused[0].consensus_status == "single_source"


def test_two_independent_families_make_high_consensus():
    fused = fuse_candidates([
        candidate("teacher-640", "reference_teacher", "fastener", (10, 10, 30, 30), 0.8),
        candidate("hsv-1", "hsv", "fastener", (12, 12, 32, 32), 0.7),
    ])
    assert fused[0].consensus_status == "consensus_high"


def test_overlapping_categories_are_conflict_not_silently_merged():
    fused = fuse_candidates([
        candidate("teacher-a", "reference_teacher", "fastener", (10, 10, 30, 30), 0.9),
        candidate("student-a", "student", "pipe_joint", (11, 11, 31, 31), 0.9),
    ])
    assert len(fused) == 1
    assert fused[0].consensus_status == "conflict"
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest ml\tests\test_auto_labeling.py -q`

Expected: collection fails because `crrc_vision.auto_labeling` does not exist.

- [ ] **Step 3: Implement the minimal candidate and fusion contract**

```python
import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    relative_path: str
    source_id: str
    source_family: str
    category: str
    xyxy: tuple[float, float, float, float]
    score: float

    def stable_id(self) -> str:
        raw = f"{self.relative_path}|{self.source_id}|{self.category}|{self.xyxy}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class FusedCandidate:
    relative_path: str
    category: str | None
    xyxy: tuple[float, float, float, float]
    member_ids: tuple[str, ...]
    supporting_families: tuple[str, ...]
    consensus_status: str


def iou_xyxy(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def fuse_candidates(rows: list[Candidate], iou_threshold: float = 0.55) -> list[FusedCandidate]:
    pending = sorted(rows, key=lambda row: (row.relative_path, row.xyxy, row.source_id))
    output = []
    while pending:
        cluster = [pending.pop(0)]
        changed = True
        while changed:
            changed = False
            keep = []
            for row in pending:
                if row.relative_path == cluster[0].relative_path and any(
                    iou_xyxy(row.xyxy, member.xyxy) >= iou_threshold for member in cluster
                ):
                    cluster.append(row)
                    changed = True
                else:
                    keep.append(row)
            pending = keep
        weight = sum(row.score for row in cluster)
        box = tuple(sum(row.xyxy[i] * row.score for row in cluster) / weight for i in range(4))
        families = tuple(sorted({row.source_family for row in cluster}))
        categories = {row.category for row in cluster}
        status = "conflict" if len(categories) > 1 else (
            "consensus_high" if len(families) > 1 else
            "propagated" if families == ("temporal",) else "single_source"
        )
        output.append(FusedCandidate(
            relative_path=cluster[0].relative_path,
            category=next(iter(categories)) if len(categories) == 1 else None,
            xyxy=box,
            member_ids=tuple(sorted(row.stable_id() for row in cluster)),
            supporting_families=families,
            consensus_status=status,
        ))
    return output
```

Implement `iou_xyxy`, deterministic clustering, score-weighted coordinates, category conflict detection, and source-family deduplication. Validate categories against `{"fastener", "pipe_joint"}`, scores in `[0, 1]`, and positive boxes.

- [ ] **Step 4: Run focused and full tests**

Run: `.\.venv\Scripts\python.exe -m pytest ml\tests\test_auto_labeling.py ml\tests -q`

Expected: focused tests pass and the full suite has no regressions.

- [ ] **Step 5: Commit**

```powershell
git add ml/src/crrc_vision/auto_labeling.py ml/tests/test_auto_labeling.py
git commit -m "feat: add safe candidate fusion contract"
```

### Task 2: Deterministic overlap tiles and inverse mapping

**Files:**
- Create: `ml/src/crrc_vision/tiles.py`
- Create: `ml/tests/test_tiles.py`

- [ ] **Step 1: Write failing tile tests**

```python
from crrc_vision.tiles import build_tiles, map_tile_box


def test_two_by_two_tiles_cover_image_with_overlap():
    tiles = build_tiles(width=2000, height=1500, grid=2, overlap=0.12)
    assert len(tiles) == 4
    assert min(t.x1 for t in tiles) == 0 and min(t.y1 for t in tiles) == 0
    assert max(t.x2 for t in tiles) == 2000 and max(t.y2 for t in tiles) == 1500
    assert tiles[0].x2 > tiles[1].x1


def test_tile_box_maps_and_clips_to_original_image():
    tile = build_tiles(2000, 1500, 2, 0.12)[3]
    mapped = map_tile_box(tile, (-10, -10, 9999, 9999), 2000, 1500)
    assert mapped == (tile.x1, tile.y1, 2000, 1500)
```

- [ ] **Step 2: Run test and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest ml\tests\test_tiles.py -q`

Expected: import failure for `crrc_vision.tiles`.

- [ ] **Step 3: Implement tiles**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Tile:
    index: int
    x1: int
    y1: int
    x2: int
    y2: int


def build_tiles(width: int, height: int, grid: int = 2, overlap: float = 0.12) -> tuple[Tile, ...]:
    if grid != 2 or not 0 <= overlap < 0.5:
        raise ValueError("supported contract is grid=2 and 0 <= overlap < 0.5")
    middle_x, middle_y = width // 2, height // 2
    expand_x, expand_y = round(width * overlap / 2), round(height * overlap / 2)
    xs = ((0, min(width, middle_x + expand_x)), (max(0, middle_x - expand_x), width))
    ys = ((0, min(height, middle_y + expand_y)), (max(0, middle_y - expand_y), height))
    return tuple(
        Tile(row * 2 + column, x1, y1, x2, y2)
        for row, (y1, y2) in enumerate(ys)
        for column, (x1, x2) in enumerate(xs)
    )


def map_tile_box(tile: Tile, xyxy: tuple[float, ...], width: int, height: int):
    x1, y1, x2, y2 = xyxy
    mapped = (
        max(0, min(width, tile.x1 + x1)),
        max(0, min(height, tile.y1 + y1)),
        max(0, min(width, tile.x1 + x2)),
        max(0, min(height, tile.y1 + y2)),
    )
    if mapped[2] <= mapped[0] or mapped[3] <= mapped[1]:
        raise ValueError("empty mapped tile box")
    return mapped
```

- [ ] **Step 4: Run focused and full tests**

Run: `.\.venv\Scripts\python.exe -m pytest ml\tests\test_tiles.py ml\tests -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add ml/src/crrc_vision/tiles.py ml/tests/test_tiles.py
git commit -m "feat: add deterministic overlap tiles"
```

### Task 3: Same-scene temporal propagation

**Files:**
- Create: `ml/src/crrc_vision/temporal.py`
- Create: `ml/tests/test_temporal.py`

- [ ] **Step 1: Write failing geometry tests**

```python
import numpy as np
import pytest
from crrc_vision.temporal import HomographyQuality, propagate_box, validate_homography
from crrc_vision.temporal import propagate_between_scenes


def test_translation_propagates_box():
    matrix = np.array([[1, 0, 5], [0, 1, 7], [0, 0, 1]], dtype=float)
    assert propagate_box((10, 20, 30, 40), matrix, 200, 100) == (15, 27, 35, 47)


def test_low_inlier_homography_is_rejected():
    quality = HomographyQuality(matches=40, inliers=7, median_error=1.0, scale=1.0)
    assert validate_homography(quality) == ("LOW_INLIER_RATIO",)


def test_cross_scene_propagation_is_rejected():
    with pytest.raises(ValueError, match="same scene"):
        propagate_between_scenes("scene-a", "scene-b", (1, 2, 3, 4), np.eye(3), 100, 100)
```

- [ ] **Step 2: Run test and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest ml\tests\test_temporal.py -q`

Expected: import failure for `crrc_vision.temporal`.

- [ ] **Step 3: Implement the temporal quality contract**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class HomographyQuality:
    matches: int
    inliers: int
    median_error: float
    scale: float


def validate_homography(q: HomographyQuality) -> tuple[str, ...]:
    errors = []
    if q.matches < 20: errors.append("TOO_FEW_MATCHES")
    if q.matches and q.inliers / q.matches < 0.35: errors.append("LOW_INLIER_RATIO")
    if q.median_error > 3.0: errors.append("HIGH_REPROJECTION_ERROR")
    if not 0.75 <= q.scale <= 1.33: errors.append("INVALID_SCALE")
    return tuple(errors)
```

Implement corner transformation with homogeneous division, bounding rectangle, clipping, and the same-scene wrapper. Keep ORB feature extraction in the script layer so geometry tests remain deterministic.

- [ ] **Step 4: Run focused and full tests**

Run: `.\.venv\Scripts\python.exe -m pytest ml\tests\test_temporal.py ml\tests -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add ml/src/crrc_vision/temporal.py ml/tests/test_temporal.py
git commit -m "feat: add guarded temporal propagation"
```

### Task 4: Codex two-pass review contract

**Files:**
- Create: `ml/src/crrc_vision/codex_review.py`
- Create: `ml/tests/test_codex_review.py`

- [ ] **Step 1: Write failing review tests**

```python
from crrc_vision.codex_review import validate_review


def sample_review(first: str, second: str | None, image_status: str = "uncertain") -> dict:
    value = {
        "reviewer": "codex-visual-auditor",
        "task_version": "safe-auto-review-v1",
        "asset_sha256": "A" * 64,
        "first_pass": {"prompt_version": "first-v1", "decision": first},
        "second_pass": None,
        "candidate_decisions": [{"candidate_id": "c1", "decision": first}],
        "added_boxes": [],
        "image_status": image_status,
        "reasons": ["visual-review"],
    }
    if second is not None:
        value["second_pass"] = {
            "prompt_version": "second-v1",
            "decision": second,
            "first_result_hidden": True,
        }
    return value


def test_adjusted_or_added_box_requires_blind_second_pass():
    review = sample_review(first="needs_adjustment", second=None)
    assert validate_review(review) == ("SECOND_PASS_REQUIRED",)


def test_uncertain_image_cannot_be_complete():
    review = sample_review(first="uncertain", second=None, image_status="complete")
    assert "UNCERTAIN_COMPLETE_CONFLICT" in validate_review(review)


def test_second_pass_must_use_distinct_prompt_and_hide_first_result():
    review = sample_review(first="needs_adjustment", second="accept")
    review["second_pass"]["prompt_version"] = review["first_pass"]["prompt_version"]
    assert "NON_INDEPENDENT_SECOND_PASS" in validate_review(review)
```

- [ ] **Step 2: Run test and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest ml\tests\test_codex_review.py -q`

Expected: import failure for `crrc_vision.codex_review`.

- [ ] **Step 3: Implement schema validation**

```python
VALID_CANDIDATE = {"accept", "reject", "needs_adjustment", "uncertain"}
VALID_IMAGE = {"complete", "uncertain"}


def _valid_normalized_box(box: object) -> bool:
    return (
        isinstance(box, list) and len(box) == 4
        and all(isinstance(value, (int, float)) and 0 <= value <= 1 for value in box)
        and box[2] > box[0] and box[3] > box[1]
    )


def validate_review(review: dict[str, object]) -> tuple[str, ...]:
    errors = set()
    decisions = review.get("candidate_decisions", [])
    values = {row.get("decision") for row in decisions}
    if not values <= VALID_CANDIDATE:
        errors.add("INVALID_CANDIDATE_DECISION")
    ids = [row.get("candidate_id") for row in decisions]
    if None in ids or len(ids) != len(set(ids)):
        errors.add("INVALID_CANDIDATE_REFERENCE")
    added = review.get("added_boxes", [])
    if any(not _valid_normalized_box(row.get("xyxy")) for row in added):
        errors.add("INVALID_ADDED_BOX")
    image_status = review.get("image_status")
    if image_status not in VALID_IMAGE:
        errors.add("INVALID_IMAGE_STATUS")
    if "uncertain" in values and image_status == "complete":
        errors.add("UNCERTAIN_COMPLETE_CONFLICT")
    requires_second = "needs_adjustment" in values or bool(added)
    second = review.get("second_pass")
    if requires_second and not second:
        errors.add("SECOND_PASS_REQUIRED")
    if second:
        first = review.get("first_pass", {})
        if (second.get("prompt_version") == first.get("prompt_version")
                or second.get("first_result_hidden") is not True):
            errors.add("NON_INDEPENDENT_SECOND_PASS")
    asset_hash = review.get("asset_sha256", "")
    if len(asset_hash) != 64 or any(ch not in "0123456789ABCDEFabcdef" for ch in asset_hash):
        errors.add("INVALID_ASSET_HASH")
    return tuple(sorted(errors))
```

Define `reviewer`, `task_version`, `asset_sha256`, `first_pass`, optional `second_pass`, `candidate_decisions`, `added_boxes`, `image_status`, and `reasons`. Reject unknown candidates and non-positive boxes.

- [ ] **Step 4: Run focused and full tests**

Run: `.\.venv\Scripts\python.exe -m pytest ml\tests\test_codex_review.py ml\tests -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add ml/src/crrc_vision/codex_review.py ml/tests/test_codex_review.py
git commit -m "feat: add Codex arbitration contract"
```

### Task 5: Whole-image silver truth gate

**Files:**
- Create: `ml/src/crrc_vision/silver_truth.py`
- Create: `ml/tests/test_silver_truth.py`

- [ ] **Step 1: Write failing gate tests**

```python
from crrc_vision.silver_truth import evaluate_image, evaluate_dataset


def sample_image(status: str, *, image_id: int = 1, split: str = "train", synthetic: bool = False):
    return {"id": image_id, "scene_group": f"g{image_id}", "split": split,
            "image_review_status": status, "synthetic": synthetic, "width": 100, "height": 100}


def sample_box(status: str, *, image_id: int = 1):
    return {"id": image_id, "image_id": image_id, "category_id": 1,
            "bbox": [10, 10, 20, 20], "review_status": status, "second_pass": "accept"}


def sample_complete_document(train_groups: int = 64, val_groups: int = 16,
                             synthetic_val: bool = False):
    images = []
    boxes = []
    for index in range(train_groups + val_groups):
        split = "train" if index < train_groups else "val"
        image = sample_image("complete", image_id=index + 1, split=split,
                             synthetic=synthetic_val and split == "val" and index == train_groups)
        images.append(image)
        boxes.append(sample_box("accept", image_id=index + 1))
    return {"images": images, "annotations": boxes,
            "categories": [{"id": 1, "name": "fastener"}, {"id": 2, "name": "pipe_joint"}]}


def test_candidate_accept_does_not_complete_image():
    report = evaluate_image(sample_image(status="uncertain"), [sample_box("accept")])
    assert not report.complete


def test_uncertain_box_excludes_whole_image():
    report = evaluate_image(sample_image(status="complete"), [sample_box("uncertain")])
    assert report.errors == ("UNRESOLVED_CANDIDATE",)


def test_dataset_requires_64_train_and_16_val_scene_groups():
    report = evaluate_dataset(sample_complete_document(train_groups=64, val_groups=15))
    assert "INSUFFICIENT_VAL_GROUPS" in report.errors


def test_synthetic_image_is_forbidden_in_val():
    report = evaluate_dataset(sample_complete_document(synthetic_val=True))
    assert "SYNTHETIC_VALIDATION_IMAGE" in report.errors
```

- [ ] **Step 2: Run test and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest ml\tests\test_silver_truth.py -q`

Expected: import failure for `crrc_vision.silver_truth`.

- [ ] **Step 3: Implement image and dataset reports**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ImageReport:
    errors: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class SilverReport:
    errors: tuple[str, ...]
    train_groups: int
    val_groups: int

    @property
    def can_train(self) -> bool:
        return not self.errors


def evaluate_image(image: dict[str, object], annotations: list[dict[str, object]]) -> ImageReport:
    errors = set()
    if image.get("image_review_status") not in {"complete", "accept_empty"}:
        errors.add("IMAGE_NOT_COMPLETE")
    if any(row.get("review_status") != "accept" for row in annotations):
        errors.add("UNRESOLVED_CANDIDATE")
    if image.get("image_review_status") == "accept_empty" and annotations:
        errors.add("BOX_ON_ACCEPTED_EMPTY_IMAGE")
    return ImageReport(tuple(sorted(errors)))


def evaluate_dataset(document: dict[str, object]) -> SilverReport:
    errors = set()
    images = document.get("images", [])
    annotations = document.get("annotations", [])
    by_image = {row["id"]: [] for row in images}
    for annotation in annotations:
        if annotation.get("image_id") not in by_image:
            errors.add("UNKNOWN_IMAGE_REFERENCE")
            continue
        by_image[annotation["image_id"]].append(annotation)
    for image in images:
        errors.update(evaluate_image(image, by_image[image["id"]]).errors)
    train = {row["scene_group"] for row in images if row.get("split") == "train"}
    val = {row["scene_group"] for row in images if row.get("split") == "val"}
    if len(train) < 64: errors.add("INSUFFICIENT_TRAIN_GROUPS")
    if len(val) < 16: errors.add("INSUFFICIENT_VAL_GROUPS")
    if train & val: errors.add("SCENE_GROUP_LEAKAGE")
    if any(row.get("synthetic") and row.get("split") == "val" for row in images):
        errors.add("SYNTHETIC_VALIDATION_IMAGE")
    return SilverReport(tuple(sorted(errors)), len(train), len(val))
```

- [ ] **Step 4: Run focused and full tests**

Run: `.\.venv\Scripts\python.exe -m pytest ml\tests\test_silver_truth.py ml\tests -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add ml/src/crrc_vision/silver_truth.py ml/tests/test_silver_truth.py
git commit -m "feat: guard AI silver truth export"
```

### Task 6: Build deterministic machine candidates

**Files:**
- Create: `ml/scripts/build_safe_auto_candidates.py`
- Modify: `ml/scripts/run_reference_teacher.py`
- Modify: `ml/tests/test_reference_teacher.py`
- Test: `ml/tests/test_auto_labeling.py`

- [ ] **Step 1: Add failing tests for pass metadata and truth immutability**

```python
def test_prediction_pass_records_scale_and_tile():
    item = TeacherPrediction(
        relative_path="a.jpg",
        teacher_class_id=2,
        teacher_class_name="class_2",
        bbox=(10.0, 20.0, 30.0, 40.0),
        score=0.9,
        pass_id="full-960",
        tile=None,
    )
    assert item.to_dict()["pass_id"] == "full-960"


def test_candidate_manifest_rejects_truth_hash_change(tmp_path):
    before = "A" * 64
    with pytest.raises(RuntimeError, match="formal truth changed"):
        verify_truth_unchanged(before, "B" * 64)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest ml\tests\test_reference_teacher.py ml\tests\test_auto_labeling.py -q`

Expected: constructor/signature failures for missing pass metadata and helper.

- [ ] **Step 3: Extend teacher predictions and add orchestration script**

The script must accept repeated `--teacher-predictions`, `--hsv-annotations`, `--manifest`, `--source`, `--truth`, and `--output`; normalize every source, run same-scene ORB/RANSAC propagation only when the Task 3 gate passes, fuse candidates, and atomically emit:

```json
{
  "schema_version": "safe-auto-candidates-v1",
  "input_hashes": {},
  "config": {"iou": 0.55, "teacher_sizes": [640, 960, 1280], "tile_overlap": 0.12},
  "images": [],
  "raw_candidates": [],
  "fused_candidates": [],
  "errors": []
}
```

Refuse an existing output directory, escaping asset paths, missing selection images, and any formal truth hash change.

- [ ] **Step 4: Run focused and full tests**

Run: `.\.venv\Scripts\python.exe -m pytest ml\tests -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add ml/scripts/build_safe_auto_candidates.py ml/scripts/run_reference_teacher.py ml/tests
git commit -m "feat: assemble safe automatic candidates"
```

### Task 7: Render Codex audit packs and merge decisions

**Files:**
- Create: `ml/src/crrc_vision/codex_review_pack.py`
- Create: `ml/scripts/build_codex_review_pack.py`
- Create: `ml/scripts/apply_codex_review.py`
- Test: `ml/tests/test_codex_review.py`

- [ ] **Step 1: Add failing pack and merge tests**

```python
import pytest
from PIL import Image
from crrc_vision.codex_review_pack import build_pack
from crrc_vision.codex_review import merge_reviews


def test_review_pack_covers_every_image_and_candidate(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (100, 100), "gray").save(source / "a.jpg")
    Image.new("RGB", (100, 100), "gray").save(source / "b.jpg")
    candidates = {
        "images": [{"id": 1, "relative_path": "a.jpg"}, {"id": 2, "relative_path": "b.jpg"}],
        "fused_candidates": [
            {"id": "c1", "image_id": 1, "xyxy": [10, 10, 30, 30]},
            {"id": "c2", "image_id": 1, "xyxy": [40, 40, 60, 60]},
            {"id": "c3", "image_id": 2, "xyxy": [20, 20, 50, 50]},
        ],
    }
    output = tmp_path / "pack"
    summary = build_pack(candidates, source, output)
    assert summary.images == 2
    assert summary.candidates == 3
    assert len(list((output / "full-images").glob("*.jpg"))) == 2


def test_merge_refuses_missing_candidate_decision():
    candidates = {"fused_candidates": [{"id": "c1"}, {"id": "c2"}]}
    incomplete = {"candidate_decisions": [{"candidate_id": "c1", "decision": "accept"}]}
    with pytest.raises(ValueError, match="missing candidate decisions"):
        merge_reviews(candidates, incomplete)
```

- [ ] **Step 2: Run test and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest ml\tests\test_codex_review.py -q`

Expected: missing pack and merge functions.

- [ ] **Step 3: Implement rendering and merge CLIs**

Render 1000-pixel review quadrants with normalized grid labels, context crops at 2× box size, same-scene neighbors, and candidate provenance. Emit first-pass tasks in batches of at most eight images. For added or adjusted boxes, emit a separate second-pass task without the first decision text. Merge only validated JSON and write an immutable decision manifest.

- [ ] **Step 4: Run focused and full tests**

Run: `.\.venv\Scripts\python.exe -m pytest ml\tests -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add ml/src/crrc_vision/codex_review_pack.py ml/scripts/build_codex_review_pack.py ml/scripts/apply_codex_review.py ml/tests/test_codex_review.py
git commit -m "feat: build two-pass Codex review packs"
```

### Task 8: Export isolated silver truth or refusal report

**Files:**
- Create: `ml/scripts/export_silver_truth.py`
- Test: `ml/tests/test_silver_truth.py`
- Modify: `docs/validation/2026-08-25-full-image-v2-bootstrap.md`
- Modify: `PROJECT_STATUS.md`

- [ ] **Step 1: Add failing CLI-level export test**

```python
from crrc_vision.silver_truth import export_silver


def test_export_writes_refusal_without_silver_coco(tmp_path):
    document = sample_complete_document(train_groups=20, val_groups=5)
    code = export_silver(document, tmp_path)
    assert code == 2
    assert (tmp_path / "silver-refusal.json").exists()
    assert not (tmp_path / "instances.silver.json").exists()
```

- [ ] **Step 2: Run test and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest ml\tests\test_silver_truth.py -q`

Expected: missing `export_silver`.

- [ ] **Step 3: Implement atomic export**

On PASS, write `instances.silver.json`, `silver-manifest.json`, accepted/uncertain image lists, source hashes, class counts, and scene-group counts. On FAIL, write only `silver-refusal.json` with stable error codes. Recompute formal truth SHA-256 before and after export and fail if it changes.

- [ ] **Step 4: Run the full Phase A pipeline on Git-external assets**

Run the pinned teacher at 640/960/1280 for full images and 2×2 tiles, then build candidates and review assets under:

`review-packs/fastener-v2/safe-auto-v1/`

Use the current Codex visual model to produce first-pass decisions for every image. Generate and review second-pass assets for every added/adjusted box. Export silver truth only if the gate passes.

Expected: all 482 source images have a machine record; formal truth hash remains `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`; either a gated silver COCO with at least 64/16 groups or an explicit refusal report exists.

- [ ] **Step 5: Verify and document evidence**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest ml\tests -q
git diff --check
git status --short
```

Record candidate counts by source, Codex accept/reject/adjust/uncertain counts, complete scene groups, output hashes, wall timing, and the exact Phase B decision. Never call AI-silver metrics production accuracy.

- [ ] **Step 6: Commit**

```powershell
git add ml/scripts/export_silver_truth.py ml/tests/test_silver_truth.py docs/validation/2026-08-25-full-image-v2-bootstrap.md PROJECT_STATUS.md
git commit -m "feat: export guarded AI silver truth"
```

## Phase B handoff gate

Create the separate PicoDet/self-training/imagegen implementation plan only when Phase A produces at least 64 complete train groups and 16 complete real val groups. If Phase A refuses, the next plan must target the dominant uncertainty bucket rather than bypassing the gate. imagegen is permitted only when that evidence identifies a real-data coverage deficit, and generated images remain capped at 20% of train and 0% of val.
