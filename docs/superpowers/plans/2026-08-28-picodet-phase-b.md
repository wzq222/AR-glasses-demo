# PicoDet Phase B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and compare PicoDet-S 416 and PicoDet-M 416 on the gated 64-train/16-val AI-silver dataset, export the better checkpoint, and run a desktop inference smoke test without modifying formal truth.

**Architecture:** A repository-owned adapter validates the silver COCO, splits it by the existing scene-level `split`, verifies every source image hash, and creates a Git-external PaddleDetection run manifest plus train/val COCO files. A guarded CLI invokes a pinned PaddleDetection checkout with model-specific overrides, then records metrics, artifacts, and immutable input hashes. Training and exported assets stay under `CRRC_VISION_DATA_ROOT`; repository tests use fake runners and never require Paddle.

**Tech Stack:** Python 3.11+, pytest, COCO JSON, PaddleDetection v2.9.0, PaddlePaddle GPU, PicoDet/PaddleClas LCNet, Paddle inference export.

---

### Task 1: Dataset and training-contract tests

**Files:**
- Create: `ml/tests/test_picodet.py`
- Create: `ml/src/crrc_vision/picodet.py`

- [ ] **Step 1: Write failing tests** for 64/16 minimums, scene leakage, invalid categories, missing or hash-mismatched images, deterministic split COCO, and immutable formal-truth verification.
- [ ] **Step 2: Run `\.venv\Scripts\python.exe -m pytest ml\tests\test_picodet.py -q`** and verify import/test failures.
- [ ] **Step 3: Implement `PicodetReadiness`, `validate_silver_dataset`, and `prepare_picodet_dataset`** with explicit error codes and deterministic JSON output.
- [ ] **Step 4: Re-run the focused tests** and require all tests to pass.

### Task 2: Reproducible external command and manifest

**Files:**
- Modify: `ml/src/crrc_vision/picodet.py`
- Modify: `ml/tests/test_picodet.py`

- [ ] **Step 1: Add failing tests** asserting the pinned checkout, S/M 416 configs, batch-scaled learning rates, fixed seed, output directory, and refusal when a dependency/config is absent.
- [ ] **Step 2: Implement command construction and manifests** so S and M share data, epochs, image size, seed, and evaluation settings while retaining model-specific official pretrained backbones.
- [ ] **Step 3: Run focused tests** and require all command/manifest tests to pass.

### Task 3: Guarded CLI

**Files:**
- Create: `ml/scripts/train_picodet.py`
- Modify: `ml/tests/test_picodet.py`

- [ ] **Step 1: Add failing CLI tests** for dry-run PASS, gate refusal exit code 2, and execute mode with a fake subprocess runner.
- [ ] **Step 2: Implement CLI arguments** for silver document, model variant, fixed PaddleDetection root/revision, epochs, batch size, prepare-only, train, evaluate, and export.
- [ ] **Step 3: Run focused tests** and require all CLI tests to pass.

### Task 4: Real S/M training and export

**Files:**
- Git-external: `runtimes/paddledetection-v2.9.0/`
- Git-external: `runs/picodet-s-v1/`
- Git-external: `runs/picodet-m-v1/`

- [ ] **Step 1: Create a dedicated Git-external Python environment**, install the official CUDA-compatible PaddlePaddle package and PaddleDetection dependencies, and run `paddle.utils.run_check()`.
- [ ] **Step 2: Clone PaddleDetection tag `v2.9.0` and record its resolved commit SHA** in both manifests.
- [ ] **Step 3: Prepare both runs from `annotations/silver-gate-cumulative-013/instances.silver.json`** and verify identical data hashes and 64/16 scene counts.
- [ ] **Step 4: Train PicoDet-S 416 and PicoDet-M 416** on one GPU with equal epoch/augmentation/evaluation policy and retain best checkpoints plus logs.
- [ ] **Step 5: Evaluate and export both best checkpoints**, recording COCO metrics, model size, export status, wall time, and any runtime failure without claiming production accuracy.

### Task 5: Desktop smoke test and handoff

**Files:**
- Create: `docs/validation/2026-08-28-picodet-phase-b.md`
- Modify: `PROJECT_STATUS.md`

- [ ] **Step 1: Run one full-image plus 2×2-overlap inference smoke test** with each successfully exported model and verify all mapped boxes remain inside the original image.
- [ ] **Step 2: Compare S and M** on AI-silver validation AP/recall, false positives per image, model size, and desktop latency; select a provisional mobile candidate without asserting phone P95.
- [ ] **Step 3: Run `\.venv\Scripts\python.exe -m pytest ml\tests -q` and `\.gradlew.bat testDebugUnitTest assembleDebug`**, then recheck formal truth SHA-256.
- [ ] **Step 4: Update validation/status documents and commit repository-only changes**, leaving field images, labels, runtimes, and weights outside Git.
