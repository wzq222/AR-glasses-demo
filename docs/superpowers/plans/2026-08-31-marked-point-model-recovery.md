# Marked-Point Model Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate a dedicated single-class marked-point YOLO-P2 proposal model, then add bounded hard-negative mining and a mobile ROI verifier only when each preceding gate passes.

**Architecture:** Reuse the guarded P2 training and sliced prediction runtime, but validate the business-specific `marked_point` COCO contract before training. Select the proposal threshold by recall and candidate burden instead of precision, then train a MobileNetV3 verifier on scene-isolated original-resolution crops. The old sealed test remains closed and all model outputs stay under `E:/crrc_vision_data`.

**Tech Stack:** Python 3.11, pytest, Ultralytics 8.2.40, PyTorch, OpenCV, COCO JSON, YOLOv8s-P2, MobileNetV3, ONNX.

---

## File boundaries

- `ml/src/crrc_vision/marked_point_training.py`: marked-point COCO and scene-isolation contract.
- `ml/src/crrc_vision/marked_point_model_gate.py`: proposal recall, threshold, burden and complete-scene metrics.
- `ml/scripts/train_p2_high_accuracy.py`: explicit marked-point experiment mode.
- `ml/scripts/evaluate_marked_point_model.py`: hash-bound proposal evaluation CLI.
- `ml/src/crrc_vision/marked_point_verifier.py`: E3 crop labels, splits and verifier metrics.
- `ml/scripts/build_marked_point_verifier_dataset.py`: E3 external crop materialization.
- `ml/tests/test_marked_point_training.py`, `test_marked_point_model_gate.py`, `test_marked_point_verifier.py`: rejection paths and metrics.

### Task 1: Freeze the marked-point training contract

**Files:**
- Create: `ml/src/crrc_vision/marked_point_training.py`
- Create: `ml/tests/test_marked_point_training.py`

- [ ] **Step 1: Write failing contract tests**

```python
from crrc_vision.marked_point_training import validate_marked_point_training_documents


def _doc(partition: str, start: int, count: int) -> dict:
    return {
        "info": {"partition": partition},
        "categories": [{"id": 1, "name": "marked_point"}],
        "images": [
            {"id": start+i, "file_name": f"{start+i}.jpg", "scene_group": f"scene-{start+i}"}
            for i in range(count)
        ],
        "annotations": [
            {"id": start+i, "image_id": start+i, "category_id": 1, "bbox": [1, 1, 10, 10]}
            for i in range(count)
        ],
    }


def test_marked_point_training_requires_one_business_class_and_scene_isolation() -> None:
    report = validate_marked_point_training_documents(_doc("train", 1, 30), _doc("val", 100, 17))
    assert report == {"train_images": 30, "train_boxes": 30, "val_images": 17, "val_boxes": 17}
```

Add exact rejection assertions for category `fastener_target`, fewer than 30/17 images, empty annotations, repeated image or annotation IDs, invalid image references and overlapping `scene_group`.

- [ ] **Step 2: Verify RED**

Run: `./.venv/Scripts/python.exe -m pytest ml/tests/test_marked_point_training.py -q`

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement the validator**

Implement `validate_marked_point_training_documents(train, val) -> dict[str, int]`. Require partitions `train/val`, exactly one category `{id: 1, name: marked_point}`, at least 30/17 images, nonempty annotations, valid annotation image references, unique IDs and disjoint scene groups.

- [ ] **Step 4: Run the focused test and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest ml/tests/test_marked_point_training.py -q
git add ml/src/crrc_vision/marked_point_training.py ml/tests/test_marked_point_training.py
git commit -m "feat: validate marked-point training inputs"
```

### Task 2: Add explicit E1 launcher mode and materialize the dataset

**Files:**
- Modify: `ml/scripts/train_p2_high_accuracy.py`
- Modify: `ml/src/crrc_vision/marked_point_training.py`
- Modify: `ml/tests/test_marked_point_training.py`
- Git-external create: `E:/crrc_vision_data/runs/marked-point-p2-e1/`

- [ ] **Step 1: Test manifest metadata**

```python
from crrc_vision.marked_point_training import experiment_manifest_fields


def test_marked_point_manifest_fields_freeze_business_target() -> None:
    assert experiment_manifest_fields() == {
        "experiment_kind": "marked_point_proposal",
        "business_target": "marked anti-loosening inspection point",
        "selection_metric": "proposal_recall_then_candidate_burden",
        "sealed_test_visible": False,
    }
```

- [ ] **Step 2: Verify RED, then implement**

Add `experiment_manifest_fields()` with the exact dictionary. Add `--experiment-kind` choices `physical_target` and `marked_point_proposal` to `train_p2_high_accuracy.py`. In marked-point mode, call Task 1 validation and merge the fields into every training manifest. Do not change model YAML, optimizer or checkpoint loading.

- [ ] **Step 3: Materialize without training**

```powershell
$env:CRRC_VISION_DATA_ROOT='E:\crrc_vision_data'
.\.venv\Scripts\python.exe ml\scripts\train_p2_high_accuracy.py `
  --experiment-kind marked_point_proposal `
  --train-coco annotations/marked-point-v1.4/instances.train.json `
  --val-coco annotations/marked-point-v1.4/instances.val.json `
  --pretrained runs/yolov8s-p2-v3-640-direct/train/weights/best.pt `
  --output runs/marked-point-p2-e1 `
  --epochs 40 --batch-size 2 --seed 20260829
```

Expected: 30 real train scenes plus deterministic train tiles, 17 untouched val scenes, one class, no sealed assets and one ready seed manifest.

- [ ] **Step 4: Verify truth and commit**

Formal truth must remain `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`.

### Task 3: Implement a proposal-specific validation gate

**Files:**
- Create: `ml/src/crrc_vision/marked_point_model_gate.py`
- Create: `ml/scripts/evaluate_marked_point_model.py`
- Create: `ml/tests/test_marked_point_model_gate.py`

- [ ] **Step 1: Write failing threshold test**

```python
from crrc_vision.marked_point_model_gate import select_proposal_threshold


def test_gate_selects_highest_threshold_that_keeps_required_recall() -> None:
    truth = {"images": [{"id": 1}, {"id": 2}], "annotations": [
        {"id": 1, "image_id": 1, "bbox": [0, 0, 10, 10]},
        {"id": 2, "image_id": 2, "bbox": [0, 0, 10, 10]},
    ]}
    predictions = [
        {"image_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 2, "bbox": [0, 0, 10, 10], "score": 0.4},
        {"image_id": 2, "bbox": [20, 20, 10, 10], "score": 0.8},
    ]
    report = select_proposal_threshold(predictions, truth, minimum_recall=1.0)
    assert report.threshold == 0.4
    assert report.recall == 1.0
    assert report.candidates_per_image == 1.5
    assert report.complete_scenes == 2
```

- [ ] **Step 2: Verify RED, then implement matching**

Use deterministic score ordering and one-to-one IoU 0.50 matching. The report fields are `threshold`, `recall`, `precision`, `candidates_per_image`, `complete_scenes`, `images`, `true_positives`, `false_positives`, and `false_negatives`. Test empty predictions, ties, duplicates and no reachable threshold.

- [ ] **Step 3: Implement hash-bound CLI**

`evaluate_marked_point_model.py` reads val truth and predictions, rejects `sealed_test`, selects at minimum recall 0.99, and writes truth/prediction/model hashes. It must never choose by AP alone.

- [ ] **Step 4: Run focused/full tests and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest ml/tests/test_marked_point_model_gate.py ml/tests/test_high_accuracy_gate.py -q
git add ml/src/crrc_vision/marked_point_model_gate.py ml/scripts/evaluate_marked_point_model.py ml/tests/test_marked_point_model_gate.py
git commit -m "feat: gate marked-point proposals by recall"
```

### Task 4: Train and evaluate the bounded E1 pilot

**Files:**
- Git-external modify: `E:/crrc_vision_data/runs/marked-point-p2-e1/seed-20260829/`
- Create: `docs/validation/2026-08-31-marked-point-p2-e1.md`

- [ ] **Step 1: Execute one 40-epoch seed**

Run the Task 2 command with `--execute`. Use batch 2 because the RTX 3060 currently has limited free VRAM. Do not terminate unrelated user applications; if CUDA OOM occurs, use a new output with batch 1.

- [ ] **Step 2: Predict full and fused validation outputs**

Run `predict_yolo_sliced.py` twice using the best checkpoint, marked-point val COCO, source root, `imgsz=640`, `conf=0.001`, and modes `full` and `fused`.

- [ ] **Step 3: Run the proposal gate**

E1 is promising only when fused recall is at least 0.99 and candidate burden is no more than 20/image. Otherwise record exact misses and do not train two more seeds.

- [ ] **Step 4: Expand to three seeds only after the pilot gate**

Use identical arguments for seeds 20260828 and 20260830. Select the lowest mean candidate burden among configurations whose worst-seed recall is at least 0.99 and recall range is at most 0.03.

### Task 5: Run E2 hard-negative mining only after E1 evidence

**Files:**
- Create: `ml/src/crrc_vision/marked_point_hard_negatives.py`
- Create: `ml/scripts/build_marked_point_hard_negatives.py`
- Create: `ml/tests/test_marked_point_hard_negatives.py`
- Git-external create: `E:/crrc_vision_data/runs/marked-point-p2-e2/`

- [ ] **Step 1: Test deterministic selection**

Select high-score false positives and near-miss positives below the threshold, with at most two per pHash/scene cluster. Exclude val and every sealed identity.

- [ ] **Step 2: Implement and materialize**

Keep original full train scenes; add only hash-bound negative crop/tile views. Train one identical seed and continue only if recall stays at least 0.99 and candidate burden improves by at least 10%.

- [ ] **Step 3: Expand to three seeds only after the E2 pilot gate**

Reject E2 if recall range exceeds 0.03 or improvement exists in only one seed.

### Task 6: Train the E3 MobileNetV3 ROI verifier

**Files:**
- Create: `ml/src/crrc_vision/marked_point_verifier.py`
- Create: `ml/scripts/build_marked_point_verifier_dataset.py`
- Create: `ml/scripts/train_marked_point_verifier.py`
- Create: `ml/tests/test_marked_point_verifier.py`
- Git-external create: `E:/crrc_vision_data/runs/marked-point-verifier-e3/`

- [ ] **Step 1: Freeze four labels and scene groups**

Labels are `marked_point`, `unmarked_fastener`, `lookalike`, `insufficient`. Use original-resolution 1.6x context crops; source scene group determines train/val.

- [ ] **Step 2: Train Small and one Large challenger**

Use ImageNet initialization, class-balanced focal loss, 224 input, three seeds and early stopping on marked-point recall under a precision constraint. Stop Large unless it improves mean recall by at least 0.02.

- [ ] **Step 3: Evaluate the full pipeline**

Pass only when final recall is at least 0.95, precision at least 0.90, complete-scene rate at least 0.90 and all `insufficient` candidates remain visible for human review.

- [ ] **Step 4: Export only after the real-val gate**

Export ONNX and benchmark the designated phone over 50 warm runs. Desktop timing is diagnostic only.

