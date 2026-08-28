# Marked-Point Candidate Recall Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broad physical-fastener target with an auditable marked-inspection-point truth set and prove that the union of color-mark and low-threshold fastener proposals reaches at least 99% candidate recall before training a verifier or status model.

**Architecture:** Build a new Git-external development selection that excludes every old sealed-test image, generate independent A/B proposal branches, fuse them without score-based deletion, and review complete full images under a marked-point-specific contract. Evaluate proposal recall only on the corrected validation truth; ordinary fasteners remain hard negatives and never enter the business recall denominator.

**Tech Stack:** Python 3.11, pytest, OpenCV, NumPy, COCO-style JSON, existing P2 predictions and safe review-pack infrastructure, Git-external CRRC asset store.

---

## Scope and file map

This plan implements Slice 1 only. It does not train the ROI verifier, build the status segmentation model,
change Android, open any sealed test, or claim production accuracy.

- `ml/src/crrc_vision/marked_point.py`: labels, review decisions, integrity validation, and positive-truth assembly.
- `ml/src/crrc_vision/marked_point_selection.py`: deterministic train/validation selection with old-sealed exclusion.
- `ml/src/crrc_vision/mark_proposals.py`: relaxed red/yellow color-mark proposals with mark geometry retained.
- `ml/src/crrc_vision/marked_point_candidates.py`: A/B normalization, union, and source-preserving deduplication.
- `ml/src/crrc_vision/candidate_recall.py`: recall, complete-scene rate, and proposal-burden metrics.
- `ml/scripts/build_marked_point_selection.py`: writes the Git-external development selection.
- `ml/scripts/build_mark_proposals.py`: runs A-branch proposals on selected full images.
- `ml/scripts/build_marked_point_candidates.py`: combines A proposals with existing B-branch candidates.
- `ml/scripts/build_marked_point_review_pack.py`: creates full-image, scan-tile, and candidate-context review assets.
- `ml/scripts/assemble_marked_point_truth.py`: validates reviews and writes separate train/val truth.
- `ml/scripts/evaluate_candidate_recall.py`: applies the 99% gate without reading sealed data.
- Tests mirror each module under `ml/tests/`.
- Private outputs stay under Git-external `selections/marked-point-v1`, `runs/marked-point-proposals-v1`,
  `review-packs/marked-point-v1`, and `annotations/marked-point-v1`.

### Task 1: Define the marked-point truth contract

**Files:**
- Create: `ml/tests/test_marked_point.py`
- Create: `ml/src/crrc_vision/marked_point.py`

- [ ] **Step 1: Write the failing contract tests**

```python
from crrc_vision.marked_point import assemble_marked_point_truth, validate_review


def test_only_marked_points_enter_positive_truth():
    review = {
        "partition": "val",
        "images": [{
            "image_id": 1,
            "relative_path": "a.jpg",
            "scene_group": "scene-a",
            "source_sha256": "A" * 64,
            "image_status": "complete",
            "expected_candidate_ids": ["m", "u", "l"],
            "candidate_decisions": [
                {"candidate_id": "m", "label": "marked_point", "xyxy": [1, 2, 11, 12]},
                {"candidate_id": "u", "label": "unmarked_fastener", "xyxy": [20, 20, 30, 30]},
                {"candidate_id": "l", "label": "lookalike", "xyxy": [40, 40, 50, 50]},
            ],
            "added_marked_points": [],
        }],
    }
    truth = assemble_marked_point_truth(review, image_sizes={"a.jpg": (100, 80)})
    assert len(truth["annotations"]) == 1
    assert truth["categories"] == [{"id": 1, "name": "marked_point"}]


def test_complete_image_rejects_uncertain_or_missing_candidates():
    review = {
        "partition": "val",
        "images": [{
            "image_id": 1,
            "relative_path": "a.jpg",
            "scene_group": "scene-a",
            "source_sha256": "A" * 64,
            "image_status": "complete",
            "expected_candidate_ids": ["a", "b"],
            "candidate_decisions": [
                {"candidate_id": "a", "label": "uncertain", "xyxy": [1, 1, 2, 2]},
            ],
            "added_marked_points": [],
        }],
    }
    errors = validate_review(review)
    assert "CANDIDATE_COVERAGE_MISMATCH:1" in errors
    assert "UNCERTAIN_COMPLETE_CONFLICT:1" in errors
```

- [ ] **Step 2: Run the focused test and verify the import failure**

Run: `.venv\Scripts\python.exe -m pytest ml\tests\test_marked_point.py -q`

Expected: collection fails because `crrc_vision.marked_point` does not exist.

- [ ] **Step 3: Implement strict labels and validation**

```python
VALID_LABELS = {
    "marked_point", "unmarked_fastener", "lookalike", "uncertain"
}


def validate_review(document: dict[str, object]) -> list[str]:
    errors: list[str] = []
    for image in document.get("images", []):
        image_id = image["image_id"]
        expected = set(image.get("expected_candidate_ids", []))
        decisions = image.get("candidate_decisions", [])
        observed = [row.get("candidate_id") for row in decisions]
        if set(observed) != expected or len(observed) != len(set(observed)):
            errors.append(f"CANDIDATE_COVERAGE_MISMATCH:{image_id}")
        labels = [row.get("label") for row in decisions]
        if any(label not in VALID_LABELS for label in labels):
            errors.append(f"INVALID_MARKED_POINT_LABEL:{image_id}")
        if image.get("image_status") == "complete" and "uncertain" in labels:
            errors.append(f"UNCERTAIN_COMPLETE_CONFLICT:{image_id}")
    return sorted(set(errors))
```

Implement `assemble_marked_point_truth` so only `marked_point` decisions and accepted
`added_marked_points` become category 1 COCO annotations. Reject invalid boxes, unknown images,
duplicate candidate IDs, duplicate scenes, incomplete images, and source hashes that differ from the
selection manifest. Store every negative decision in `info.negative_counts`, not as positive COCO boxes.

- [ ] **Step 4: Run the focused tests**

Run: `.venv\Scripts\python.exe -m pytest ml\tests\test_marked_point.py -q`

Expected: all marked-point contract tests pass.

- [ ] **Step 5: Commit the contract**

```powershell
git add ml/tests/test_marked_point.py ml/src/crrc_vision/marked_point.py
git commit -m "feat(vision): define marked-point truth contract"
```

### Task 2: Freeze a development-only selection

**Files:**
- Create: `ml/tests/test_marked_point_selection.py`
- Create: `ml/src/crrc_vision/marked_point_selection.py`
- Create: `ml/scripts/build_marked_point_selection.py`

- [ ] **Step 1: Write isolation tests**

```python
from crrc_vision.marked_point_selection import build_marked_point_selection


def test_selection_contains_all_val_and_no_old_sealed_rows():
    result = build_marked_point_selection(
        train_rows=[{"image_id": 1, "scene_group": "t1", "sha256": "1" * 64,
                     "relative_path": "t.jpg", "brightness": 90, "focus_score": 30,
                     "fused_candidate_count": 2}],
        val_rows=[{"image_id": 2, "scene_group": "v1", "sha256": "2" * 64,
                   "relative_path": "v.jpg", "brightness": 80, "focus_score": 40,
                   "fused_candidate_count": 3}],
        old_sealed_hashes={"3" * 64},
        train_count=1,
        seed=20260828,
    )
    assert [row["scene_group"] for row in result["val"]] == ["v1"]
    assert not ({row["sha256"] for row in result["train"] + result["val"]}
                & {"3" * 64})
    assert result["old_sealed_test_opened"] is False
```

Add cases rejecting a path, SHA-256, image ID, or scene shared with old sealed-test; rejecting fewer
than 19 validation scenes in the real CLI; and producing byte-identical JSON from reordered inputs.

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv\Scripts\python.exe -m pytest ml\tests\test_marked_point_selection.py -q`

Expected: import failure for `crrc_vision.marked_point_selection`.

- [ ] **Step 3: Implement deterministic selection**

Select every current high-accuracy-v2 validation scene and 40 train scenes. Stratify train by brightness,
focus, fused-candidate-count quartiles, and dominant error bucket when present. Within each stratum order by
SHA-256 of `"20260828|marked-point-v1|{scene_group}"`; round-robin to 40 independent train scenes.
The output contains `schema_version`, `seed`, `train`, `val`, forbidden old-sealed scene/path/hash sets,
`old_sealed_test_opened=false`, input hashes, and formal-truth SHA-256.

- [ ] **Step 4: Run the guarded CLI**

```powershell
$env:CRRC_VISION_DATA_ROOT='E:\crrc_vision_data'
.\.venv\Scripts\python.exe ml\scripts\build_marked_point_selection.py `
  --partition selections/high-accuracy-v2/partition.json `
  --error-pack review-packs/high-accuracy-errors-v2/errors.json `
  --truth annotations/fastener-v2/instances.json `
  --output selections/marked-point-v1/selection.json
```

Expected: `train_scenes=40`, `val_scenes=19`, `old_sealed_overlap=0`. Refuse a non-empty output,
verify formal truth before/after, and write atomically.

- [ ] **Step 5: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest ml\tests\test_marked_point_selection.py ml\tests -q
git add ml/tests/test_marked_point_selection.py ml/src/crrc_vision/marked_point_selection.py ml/scripts/build_marked_point_selection.py
git commit -m "feat(vision): freeze marked-point development selection"
```

### Task 3: Build the A-branch color-mark proposals

**Files:**
- Create: `ml/tests/test_mark_proposals.py`
- Create: `ml/src/crrc_vision/mark_proposals.py`
- Create: `ml/scripts/build_mark_proposals.py`

- [ ] **Step 1: Write synthetic-image proposal tests**

```python
import cv2
import numpy as np
from crrc_vision.mark_proposals import find_color_mark_proposals


def test_red_and_yellow_marks_are_kept_with_geometry():
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.line(image, (50, 100), (95, 100), (0, 0, 210), 5)
    cv2.line(image, (250, 200), (295, 210), (0, 210, 210), 5)
    proposals = find_color_mark_proposals(image, minimum_area=8)
    assert {row.color for row in proposals} == {"red", "yellow"}
    assert all(row.mark_xyxy[2] > row.mark_xyxy[0] for row in proposals)
    assert all(row.roi_xyxy[2] > row.roi_xyxy[0] for row in proposals)


def test_nearby_mark_fragments_are_not_dropped_by_roi_overlap():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.line(image, (70, 90), (95, 90), (0, 0, 220), 4)
    cv2.line(image, (105, 90), (130, 90), (0, 0, 220), 4)
    proposals = find_color_mark_proposals(image, minimum_area=5)
    assert len(proposals) == 2
```

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv\Scripts\python.exe -m pytest ml\tests\test_mark_proposals.py -q`

Expected: import failure for `crrc_vision.mark_proposals`.

- [ ] **Step 3: Implement mark-level proposals**

Create immutable `ColorMarkProposal` fields `color`, `mark_xyxy`, `roi_xyxy`, `line_xyxy`, `area`,
`elongation`, and `score`. Use relaxed HSV ranges for dark/bright red and yellow plus Lab chroma support.
Retain each connected mark fragment before ROI expansion. Expand ROI side to
`clamp(6 * mark_long_axis, 96, 320)` and clip to the original image. Deduplicate only mark masks with
IoU `>= 0.80`; never merge based on expanded ROI overlap.

- [ ] **Step 4: Run the selected-image CLI**

```powershell
$env:CRRC_VISION_DATA_ROOT='E:\crrc_vision_data'
.\.venv\Scripts\python.exe ml\scripts\build_mark_proposals.py `
  --selection selections/marked-point-v1/selection.json `
  --source source/20240529-luosi `
  --truth annotations/fastener-v2/instances.json `
  --output runs/marked-point-proposals-v1/a-color/proposals.json
```

Record per-image source hash, every mark and ROI geometry, zero-proposal images, thresholds, code commit,
and formal-truth before/after hash.

- [ ] **Step 5: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest ml\tests\test_mark_proposals.py ml\tests -q
git add ml/tests/test_mark_proposals.py ml/src/crrc_vision/mark_proposals.py ml/scripts/build_mark_proposals.py
git commit -m "feat(vision): generate high-recall color-mark proposals"
```

### Task 4: Union A/B proposals without score-based loss

**Files:**
- Create: `ml/tests/test_marked_point_candidates.py`
- Create: `ml/src/crrc_vision/marked_point_candidates.py`
- Create: `ml/scripts/build_marked_point_candidates.py`

- [ ] **Step 1: Write source-preservation tests**

```python
from crrc_vision.marked_point_candidates import Proposal, union_proposals


def test_a_only_and_b_only_candidates_both_survive():
    rows = [
        Proposal("a.jpg", "a1", "color_mark", (10, 10, 30, 30), 0.2),
        Proposal("a.jpg", "b1", "fastener_p2", (100, 100, 130, 130), 0.01),
    ]
    fused = union_proposals(rows, iou_threshold=0.60)
    assert len(fused) == 2
    assert {tuple(row.sources) for row in fused} == {
        ("color_mark",), ("fastener_p2",)
    }


def test_overlap_retains_both_sources():
    rows = [
        Proposal("a.jpg", "a1", "color_mark", (10, 10, 40, 40), 0.2),
        Proposal("a.jpg", "b1", "fastener_p2", (11, 11, 41, 41), 0.01),
    ]
    fused = union_proposals(rows, iou_threshold=0.60)
    assert len(fused) == 1
    assert fused[0].sources == ("color_mark", "fastener_p2")
    assert fused[0].member_ids == ("a1", "b1")
```

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv\Scripts\python.exe -m pytest ml\tests\test_marked_point_candidates.py -q`

Expected: import failure.

- [ ] **Step 3: Implement deterministic union**

Normalize A proposals from Task 3 and B proposals from
`runs/safe-auto-candidates-v2.2/candidates.json`. Filter both sources to the 59 selected paths and reject any
old-sealed path/hash. Keep every B fused candidate regardless of category or consensus. Cluster only within the
same image at IoU `>= 0.60`; use complete-link clustering so adjacent marked points cannot bridge. Store union
box, member IDs, sorted sources, and every A-member color-mark geometry.

- [ ] **Step 4: Run the union CLI**

```powershell
$env:CRRC_VISION_DATA_ROOT='E:\crrc_vision_data'
.\.venv\Scripts\python.exe ml\scripts\build_marked_point_candidates.py `
  --selection selections/marked-point-v1/selection.json `
  --color-proposals runs/marked-point-proposals-v1/a-color/proposals.json `
  --fastener-candidates runs/safe-auto-candidates-v2.2/candidates.json `
  --truth annotations/fastener-v2/instances.json `
  --output runs/marked-point-proposals-v1/union/candidates.json
```

Expected: all 59 images present, source histogram recorded, old-sealed overlap zero, formal truth unchanged.

- [ ] **Step 5: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest ml\tests\test_marked_point_candidates.py ml\tests -q
git add ml/tests/test_marked_point_candidates.py ml/src/crrc_vision/marked_point_candidates.py ml/scripts/build_marked_point_candidates.py
git commit -m "feat(vision): union marked-point proposal branches"
```

### Task 5: Build a marked-point-specific two-pass review pack

**Files:**
- Create: `ml/tests/test_marked_point_review_pack.py`
- Create: `ml/src/crrc_vision/marked_point_review_pack.py`
- Create: `ml/scripts/build_marked_point_review_pack.py`

- [ ] **Step 1: Write pack completeness and seal tests**

```python
import pytest
from crrc_vision.marked_point_review_pack import build_review_pack


def test_pack_has_full_image_four_scans_and_every_candidate(tmp_path):
    summary = build_review_pack(selection(), candidates(), source_root(), tmp_path)
    assert summary.images == 2
    assert summary.scan_tiles == 8
    assert summary.candidates == 3
    assert all(row["business_target"] == "marked anti-loosening inspection point"
               for row in load_tasks(tmp_path))


def test_pack_refuses_old_sealed_hash(tmp_path):
    with pytest.raises(ValueError, match="OLD_SEALED_IMAGE_FORBIDDEN"):
        build_review_pack(selection_with_sealed_hash(), candidates(), source_root(), tmp_path)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv\Scripts\python.exe -m pytest ml\tests\test_marked_point_review_pack.py -q`

Expected: import failure.

- [ ] **Step 3: Implement review assets and contract**

For every selected image create one 1600-pixel whole-image view, four overlapping full-resolution scan tiles,
and one context crop per fused candidate. Include every candidate ID exactly once and these reviewer rules:

```json
{
  "positive": "A fastening or pipe-joint inspection point carrying an intentional red/yellow anti-loosening mark",
  "unmarked_fastener": "A real fastener or pipe joint without an intentional anti-loosening mark",
  "lookalike": "Background, rust, sticker, reflection, wire, hole, or unrelated painted structure",
  "uncertain": "Pixels cannot prove target identity or intentional mark ownership",
  "full_image_rule": "Scan all four tiles and add every independently boundable missed marked point"
}
```

The first pass may add `added_marked_points`. Any added or geometrically adjusted positive must enter a second
pass whose task hides the first-pass label and exposes only image geometry. A complete image cannot retain an
uncertain decision.

- [ ] **Step 4: Build the Git-external pack**

```powershell
$env:CRRC_VISION_DATA_ROOT='E:\crrc_vision_data'
.\.venv\Scripts\python.exe ml\scripts\build_marked_point_review_pack.py `
  --selection selections/marked-point-v1/selection.json `
  --candidates runs/marked-point-proposals-v1/union/candidates.json `
  --source source/20240529-luosi `
  --output review-packs/marked-point-v1
```

Expected: 59 images, 236 scan tiles, every union candidate represented once, old-sealed overlap zero.

- [ ] **Step 5: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest ml\tests\test_marked_point_review_pack.py ml\tests -q
git add ml/tests/test_marked_point_review_pack.py ml/src/crrc_vision/marked_point_review_pack.py ml/scripts/build_marked_point_review_pack.py
git commit -m "feat(vision): build marked-point review pack"
```

### Task 6: Complete full-image review and assemble corrected truth

**Files:**
- Create: `ml/scripts/assemble_marked_point_truth.py`
- Modify: `ml/tests/test_marked_point.py`

- [ ] **Step 1: Add assembly integrity tests**

Add cases proving train/val scene isolation, exact candidate coverage, second-pass requirement for added boxes,
rejection of old-sealed hashes, rejection of uncertain images from complete truth, and preservation of formal truth.

```python
def test_added_positive_requires_blind_geometry_second_pass():
    review = complete_review_with_added_box(second_pass=None)
    with pytest.raises(ValueError, match="SECOND_PASS_REQUIRED"):
        assemble_partition(review, partition="train")
```

- [ ] **Step 2: Run the new test and verify failure**

Run: `.venv\Scripts\python.exe -m pytest ml\tests\test_marked_point.py -q`

Expected: fails because the partition assembler and second-pass guard are absent.

- [ ] **Step 3: Implement the guarded assembler CLI**

The CLI accepts the selection and review, verifies every pack/input hash, calls the Task 1 assembler, checks zero
train/val/old-sealed scene/path/hash overlap, and atomically writes:

```text
annotations/marked-point-v1/instances.train.json
annotations/marked-point-v1/instances.val.json
annotations/marked-point-v1/negatives.json
annotations/marked-point-v1/manifest.json
```

The manifest records positive count, negative-class counts, uncertain exclusions, complete-scene counts, hashes,
formal-truth before/after hash, and `old_sealed_test_opened=false`.

- [ ] **Step 4: Perform the complete Codex review**

Review all 59 images in fixed batches. For each image inspect the whole image and all four scan tiles before
closing candidate decisions. Label ordinary fasteners `unmarked_fastener`, not positive. Add every missed marked
inspection point. Run hidden-label geometry second pass for every added or adjusted positive. Keep genuinely
ambiguous images excluded as `uncertain`; do not lower truth quality to meet the recall gate.

- [ ] **Step 5: Assemble and inspect counts**

```powershell
$env:CRRC_VISION_DATA_ROOT='E:\crrc_vision_data'
.\.venv\Scripts\python.exe ml\scripts\assemble_marked_point_truth.py `
  --selection selections/marked-point-v1/selection.json `
  --review review-packs/marked-point-v1/review-complete.json `
  --truth annotations/fastener-v2/instances.json `
  --output annotations/marked-point-v1
```

Expected: no overlap, no unresolved decisions in included scenes, at least one positive in val, ordinary
fasteners absent from positive annotations, and formal truth unchanged.

- [ ] **Step 6: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest ml\tests\test_marked_point.py ml\tests -q
git add ml/tests/test_marked_point.py ml/scripts/assemble_marked_point_truth.py
git commit -m "feat(vision): assemble marked-point development truth"
```

### Task 7: Enforce the 99% candidate-recall gate

**Files:**
- Create: `ml/tests/test_candidate_recall.py`
- Create: `ml/src/crrc_vision/candidate_recall.py`
- Create: `ml/scripts/evaluate_candidate_recall.py`

- [ ] **Step 1: Write exact matching and completeness tests**

```python
from crrc_vision.candidate_recall import evaluate_candidate_recall


def test_candidate_gate_ignores_proposal_precision_but_requires_every_target():
    truth = coco(images=[image(1)], boxes=[box(1, 10, 10, 20, 20)])
    candidates = [candidate(1, 10, 10, 20, 20), candidate(1, 60, 60, 10, 10)]
    report = evaluate_candidate_recall(candidates, truth, minimum_recall=0.99)
    assert report.recall == 1.0
    assert report.complete_scene_rate == 1.0
    assert report.candidates_per_image == 2.0
    assert report.passed


def test_duplicate_candidate_cannot_match_two_truth_boxes():
    truth = coco(images=[image(1)], boxes=[
        box(1, 10, 10, 20, 20), box(1, 30, 10, 20, 20)
    ])
    report = evaluate_candidate_recall([candidate(1, 10, 10, 20, 20)], truth)
    assert report.matched_targets == 1
    assert report.missed_targets == 1
```

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv\Scripts\python.exe -m pytest ml\tests\test_candidate_recall.py -q`

Expected: import failure.

- [ ] **Step 3: Implement the pure gate**

Use deterministic one-to-one matching at IoU 0.50. Report total/matched/missed targets, recall, complete scenes,
complete-scene rate, candidate count, candidates per image, A-only/B-only/both coverage, target-size bands, and a
stable missed-truth list. Pass only when recall `>= 0.99` and complete-scene rate `>= 0.95`. Candidate precision
is deliberately not a pass condition in Slice 1.

- [ ] **Step 4: Run the validation-only evaluator**

```powershell
$env:CRRC_VISION_DATA_ROOT='E:\crrc_vision_data'
.\.venv\Scripts\python.exe ml\scripts\evaluate_candidate_recall.py `
  --truth annotations/marked-point-v1/instances.val.json `
  --candidates runs/marked-point-proposals-v1/union/candidates.json `
  --selection selections/marked-point-v1/selection.json `
  --output runs/marked-point-proposals-v1/union/val-recall.json
```

Refuse any path or partition metadata containing `sealed`, verify prediction/truth/selection hashes, and write
the exact missed-target list. Exit 0 on PASS and exit 2 on FAIL.

- [ ] **Step 5: Apply the stop decision**

If PASS, freeze proposal configuration and write the Slice 2 ROI-verifier plan. If FAIL, generate a miss-only
pack grouped by color, size, darkness, blur, and A/B source coverage; change only the missing proposal source and
re-run this gate. Do not start ROI or state training below 0.99 candidate recall.

- [ ] **Step 6: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest ml\tests\test_candidate_recall.py ml\tests -q
git add ml/tests/test_candidate_recall.py ml/src/crrc_vision/candidate_recall.py ml/scripts/evaluate_candidate_recall.py
git commit -m "feat(vision): enforce marked-point candidate recall gate"
```

### Task 8: Close Slice 1 with reproducible evidence

**Files:**
- Create: `docs/validation/2026-08-28-marked-point-candidate-recall.md`
- Modify: `PROJECT_STATUS.md`
- Modify: `README.md`

- [ ] **Step 1: Run the final verification suite**

```powershell
.\.venv\Scripts\python.exe -m pytest ml\tests -q
.\gradlew.bat testDebugUnitTest assembleDebug
git diff --check
Get-FileHash 'E:\Work\京新数智\识动hicool\中车眼镜数据资产\annotations\fastener-v2\instances.json' -Algorithm SHA256
git status --short
```

Expected: all tests and Android build pass; formal truth equals
`B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`; no image, label,
prediction, review, or weight file is staged; old sealed-test remains unopened.

- [ ] **Step 2: Write the validation report**

Record selected and complete train/val scenes, marked points, each negative class, uncertain exclusions,
A-only/B-only/both candidates, recall by size and error bucket, complete-scene rate, candidates per image,
missed IDs, hashes, formal-truth hash, sealed status, and PASS/FAIL. State that this is proposal recall, not final
marked-point precision or looseness accuracy.

- [ ] **Step 3: Update durable project state**

Set `PROJECT_STATUS.md` Active Work and Next Smallest Action from the measured gate. Add the validation report
to the README index. Preserve the old high-accuracy failure evidence.

- [ ] **Step 4: Commit the Slice 1 evidence**

```powershell
git add docs/validation/2026-08-28-marked-point-candidate-recall.md PROJECT_STATUS.md README.md
git commit -m "docs(vision): record marked-point candidate recall gate"
```
