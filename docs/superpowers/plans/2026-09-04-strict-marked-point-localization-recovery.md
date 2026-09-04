# Strict Marked-point Localization Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and strictly evaluate one full-tile marked-point mobile candidate challenger without modifying formal truth.

**Architecture:** Parameterize the existing deterministic COCO-to-YOLO materializer so one experiment can use all four training tiles. Fine-tune one fixed-seed P2-S challenger on the existing tight truth plus hard negatives, then reject or promote it using the corrected one-to-one gate.

**Tech Stack:** Python 3.11, pytest, Ultralytics 8.2.40, PyTorch 2.7.1 CUDA, COCO JSON, YOLOv8 P2.

---

### Task 1: Make training tile coverage explicit

**Files:**
- Modify: `ml/scripts/train_p2_high_accuracy.py`
- Modify: `ml/tests/test_train_p2_high_accuracy.py`

- [ ] **Step 1: Write the failing contract test**

Add a parser-facing test that invokes the script with `--train-tile-views 4` in prepare-only mode and asserts the generated manifest records `train_tile_views: 4` and the dataset contains five views per source train image.

- [ ] **Step 2: Verify RED**

Run: `./.venv-ml/Scripts/python.exe -m pytest ml/tests/test_train_p2_high_accuracy.py -q`

Expected: failure because `--train-tile-views` is unknown.

- [ ] **Step 3: Implement the minimal parameter**

Add `parser.add_argument("--train-tile-views", type=int, choices=(1, 2, 3, 4), default=1)`, pass it to `prepare_yolo_dataset`, and record it in the training manifest.

- [ ] **Step 4: Verify GREEN and regression**

Run: `./.venv-ml/Scripts/python.exe -m pytest ml/tests/test_train_p2_high_accuracy.py ml/tests/test_yolo_p2.py -q`

Expected: all selected tests pass.

### Task 2: Run the isolated full-tile challenger

**Files:**
- Create outside Git: `runs/marked-point-p2-e5-full-tiles/`

- [ ] **Step 1: Verify inputs and formal hash**

Confirm the E2 hard-negative train COCO, marked-point val COCO and E2 best checkpoint exist; verify formal truth equals the pinned SHA-256.

- [ ] **Step 2: Train one fixed seed**

Run `ml/scripts/train_p2_high_accuracy.py` with experiment `marked_point_proposal`, seed `20260829`, variant `s`, `--train-tile-views 4`, fine-tune mode, 20 epochs, batch 1 and a new output directory.

- [ ] **Step 3: Preserve the complete manifest**

Require `best.pt`, `last.pt`, `results.csv`, input hashes, code commit and the four-tile parameter. A missing checkpoint or incomplete epoch record is a failed run.

### Task 3: Apply the strict localization gate

**Files:**
- Create outside Git: `runs/marked-point-p2-e5-full-tiles/evaluation/`
- Modify: `docs/validation/2026-09-04-marked-point-localization-gate-correction.md`
- Modify: `PROJECT_STATUS.md`

- [ ] **Step 1: Generate predictions**

Use `ml/scripts/predict_yolo_sliced.py` at 512 and confidence 0.001 for `full`, `sliced` and `fused` modes against the frozen 17-image val set.

- [ ] **Step 2: Evaluate with the corrected gate**

Run `ml/scripts/evaluate_marked_point_model.py` on each prediction file with one-to-one IoU 0.30 coverage and the separate IoU 0.50 metric.

- [ ] **Step 3: Compare against the frozen baseline**

Record truth hits, complete scenes, IoU 0.50 recall, candidates/image and hashes. Promote only at least 74/75 and 16/17 at IoU 0.30; otherwise preserve it as a rejected experiment.

- [ ] **Step 4: Run full verification**

Run: `./.venv-ml/Scripts/python.exe -m pytest ml/tests -q`

Run: `git diff --check`

Verify the formal truth hash again and commit only code, tests and validation documentation; never commit data or weights.

### Task 4: Gate mobile replacement

**Files:**
- Modify only after Task 3 passes: Android model asset and its integrity manifest

- [ ] **Step 1: Stop safely on gate failure**

If Task 3 is below the promotion threshold, do not export or replace the APK model; leave the installed human-review build unchanged.

- [ ] **Step 2: Export only a passing challenger**

If it passes, export FP32 NCNN, run desktop ONNX/NCNN coordinate parity, then replace the Android asset together with its pinned hash.

- [ ] **Step 3: Verify on the connected phone**

Build, install and rerun the frozen 17 images plus `LOCK-REAL-01`; record strict recall, overlap count, P50/P95 and APK SHA-256 before making any completion claim.
