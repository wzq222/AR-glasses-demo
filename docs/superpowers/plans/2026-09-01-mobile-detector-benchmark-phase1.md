# Mobile Detector Benchmark Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a guarded benchmark contract and determine whether the current YOLOv8s-P2 can be converted to ncnn and MNN without losing detections before training NanoDet-Plus and PicoDet challengers.

**Architecture:** Repository code owns deterministic contracts, command construction, parity checks, and reports. Third-party source, converted models, logs, images, and weights stay below `E:/crrc_vision_data`; each run binds the source model, validation inputs, tool revision, output artifacts, and formal-truth hash. Phase 1 stops at conversion and desktop parity; only survivors enter Android JNI benchmarking.

**Tech Stack:** Python 3.12, pytest, ONNX 1.22, ONNX Runtime 1.24, ncnn `2130e00c6efd910d3e926867ca94a2d96eaf9d31`, MNN `47a656efa06ba24937e800719ecbc2165806191e`, CMake 3.31.6, Ninja, Windows 11.

---

### Task 1: Add the recall-first candidate gate

**Files:**
- Create: `ml/src/crrc_vision/mobile_benchmark.py`
- Create: `ml/tests/test_mobile_benchmark.py`

- [ ] **Step 1: Write the failing tests**

```python
from crrc_vision.mobile_benchmark import BenchmarkGate, CandidateMetrics, evaluate_candidate


def test_rejects_fast_candidate_when_recall_drops() -> None:
    gate = BenchmarkGate(0.584, 0.0, 0.60, 500.0, 2.0, 250.0)
    metrics = CandidateMetrics(0.55, 0.90, 0.10, 100.0, 8.0, 100.0)
    result = evaluate_candidate(metrics, gate)
    assert result.passed is False
    assert result.reasons == ("RECALL_BELOW_BASELINE",)


def test_accepts_only_when_accuracy_and_hot_gates_pass() -> None:
    gate = BenchmarkGate(0.584, 0.0, 0.60, 500.0, 2.0, 250.0)
    metrics = CandidateMetrics(0.60, 0.65, 0.10, 420.0, 2.2, 220.0)
    result = evaluate_candidate(metrics, gate)
    assert result.passed is True
```

- [ ] **Step 2: Verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest ml\tests\test_mobile_benchmark.py -q`

Expected: collection fails because the module is missing.

- [ ] **Step 3: Implement immutable metrics and stable reasons**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkGate:
    baseline_recall: float
    baseline_complete_scene_recall: float
    minimum_precision: float
    maximum_hot_p95_ms: float
    minimum_hot_fps: float
    maximum_pss_mb: float


@dataclass(frozen=True)
class CandidateMetrics:
    recall: float
    precision: float
    complete_scene_recall: float
    hot_p95_ms: float
    hot_fps: float
    pss_mb: float


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: tuple[str, ...]


def evaluate_candidate(metrics: CandidateMetrics, gate: BenchmarkGate) -> GateResult:
    reasons: list[str] = []
    if metrics.recall < gate.baseline_recall - 0.01:
        reasons.append("RECALL_BELOW_BASELINE")
    if metrics.precision < gate.minimum_precision:
        reasons.append("PRECISION_BELOW_MINIMUM")
    if metrics.complete_scene_recall < gate.baseline_complete_scene_recall:
        reasons.append("COMPLETE_SCENE_RECALL_BELOW_BASELINE")
    if metrics.hot_p95_ms >= gate.maximum_hot_p95_ms:
        reasons.append("HOT_P95_TOO_SLOW")
    if metrics.hot_fps < gate.minimum_hot_fps:
        reasons.append("HOT_FPS_TOO_LOW")
    if metrics.pss_mb > gate.maximum_pss_mb:
        reasons.append("PSS_TOO_HIGH")
    return GateResult(not reasons, tuple(reasons))
```

- [ ] **Step 4: Verify GREEN and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest ml\tests\test_mobile_benchmark.py -q
.\.venv\Scripts\python.exe -m pytest ml\tests -q
git add -- ml/src/crrc_vision/mobile_benchmark.py ml/tests/test_mobile_benchmark.py
git commit -m "feat(ml): add mobile detector benchmark gate"
```

### Task 2: Bind provenance and protected truth

**Files:**
- Modify: `ml/src/crrc_vision/mobile_benchmark.py`
- Modify: `ml/tests/test_mobile_benchmark.py`
- Create: `ml/scripts/prepare_mobile_benchmark.py`

- [ ] **Step 1: Add a failing test that requires uppercase model/truth hashes, a known runtime, and its exact pinned revision**

```python
def test_manifest_binds_model_truth_and_runtime(tmp_path) -> None:
    model = tmp_path / "model.onnx"
    truth = tmp_path / "truth.json"
    model.write_bytes(b"model")
    truth.write_bytes(b"truth")
    manifest = prepare_benchmark_manifest(
        "ncnn-fp16-cpu", model, truth, sha256(truth),
        "ncnn", PINNED_NCNN_COMMIT,
    )
    assert manifest["model_sha256"] == sha256(model)
    assert manifest["formal_truth_sha256"] == sha256(truth)
    assert manifest["runtime_revision"] == PINNED_NCNN_COMMIT
```

- [ ] **Step 2: Verify RED, then implement**

Add pinned constants for ncnn and MNN. Reject a truth mismatch, missing model, unknown runtime, or revision mismatch. The CLI resolves outputs below `CRRC_VISION_DATA_ROOT`, refuses a non-empty run directory, and atomically writes `benchmark-manifest.json`.

- [ ] **Step 3: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest ml\tests\test_mobile_benchmark.py -q
.\.venv\Scripts\python.exe -m pytest ml\tests -q
git add -- ml/src/crrc_vision/mobile_benchmark.py ml/tests/test_mobile_benchmark.py ml/scripts/prepare_mobile_benchmark.py
git commit -m "feat(ml): bind mobile benchmark provenance"
```

### Task 3: Add deterministic runtime export commands

**Files:**
- Create: `ml/src/crrc_vision/mobile_runtime_export.py`
- Create: `ml/tests/test_mobile_runtime_export.py`
- Create: `ml/scripts/export_mobile_runtime.py`

- [ ] **Step 1: Write failing tests for argument arrays and revision checks**

```python
def test_mnn_command_is_path_explicit(tmp_path) -> None:
    converter = tmp_path / "MNNConvert.exe"
    model = tmp_path / "model.onnx"
    output = tmp_path / "model.mnn"
    converter.write_bytes(b"exe")
    model.write_bytes(b"onnx")
    assert build_mnn_command(converter, model, output) == [
        str(converter.resolve()), "-f", "ONNX", "--modelFile",
        str(model.resolve()), "--MNNModel", str(output.resolve()),
        "--bizCode", "crrc-fastener",
    ]
```

- [ ] **Step 2: Verify RED, then implement command builders**

Return argument arrays for `MNNConvert`, `pnnx`, CMake configure, and CMake build. Never interpolate paths into a shell string. Validate checkout HEAD with `git -C <root> rev-parse HEAD` and require all artifacts below the requested Git-external run root.

- [ ] **Step 3: Implement the guarded CLI**

Accept `--runtime ncnn|mnn`, `--checkout`, `--tool`, `--model`, `--run`, and `--execute`. Dry-run writes commands and hashes. Execute mode records stdout/stderr, exit code, output hashes, elapsed time, and formal-truth hashes before and after.

- [ ] **Step 4: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest ml\tests\test_mobile_runtime_export.py -q
.\.venv\Scripts\python.exe -m pytest ml\tests -q
git add -- ml/src/crrc_vision/mobile_runtime_export.py ml/tests/test_mobile_runtime_export.py ml/scripts/export_mobile_runtime.py
git commit -m "feat(ml): guard mobile runtime exports"
```

### Task 4: Build converters and convert the current model outside Git

**Files:**
- Git-external: `E:/crrc_vision_data/runtimes/ncnn-2130e00/`
- Git-external: `E:/crrc_vision_data/runtimes/MNN-47a656e/`
- Git-external: `E:/crrc_vision_data/tools/ncnn-host/`
- Git-external: `E:/crrc_vision_data/tools/mnn-host/`
- Git-external: `E:/crrc_vision_data/runs/mobile-runtime-export-v1/`

- [ ] **Step 1: Clone exact revisions into new directories**

Clone with blob filtering, detach at the plan-header commits, and record HEAD. Do not modify existing PaddleDetection or Ultralytics runtimes.

- [ ] **Step 2: Build only the needed host converters**

Use `E:/Android/Sdk/cmake/3.31.6/bin/cmake.exe` and bundled Ninja to build pnnx/ncnn conversion tools and `MNNConvert`. Save configure/build logs below the external run.

- [ ] **Step 3: Convert without overwriting current artifacts**

Checkpoint: `E:/crrc_vision_data/runs/yolov8s-p2-v3-640/train/weights/best.pt`.

ONNX: `E:/crrc_vision_data/exports/android/yolov8s-p2-v3-640/fastener-target-p2-640.onnx`.

Produce ncnn FP16 and MNN FP32 artifacts below `mobile-runtime-export-v1`.

- [ ] **Step 4: Enforce structural checks**

Require non-empty models, fixed `1×3×640×640` input, six output channels, 34,000 candidates, exact source hashes, pinned converter revisions, and unchanged formal truth. Otherwise record `conversion_failed`.

### Task 5: Add prediction parity evaluation

**Files:**
- Modify: `ml/src/crrc_vision/mobile_benchmark.py`
- Modify: `ml/tests/test_mobile_benchmark.py`
- Create: `ml/scripts/evaluate_mobile_parity.py`

- [ ] **Step 1: Write failing tests for missing detections and bounded drift**

```python
def test_parity_rejects_missing_detection() -> None:
    baseline = [{"image_id": 1, "bbox": [10, 10, 20, 20], "score": 0.8}]
    result = compare_predictions(baseline, [], iou_threshold=0.95)
    assert result.passed is False
    assert result.missing == 1


def test_parity_accepts_small_numeric_drift() -> None:
    baseline = [{"image_id": 1, "bbox": [10, 10, 20, 20], "score": 0.8}]
    candidate = [{"image_id": 1, "bbox": [10.2, 9.9, 20, 20], "score": 0.795}]
    assert compare_predictions(baseline, candidate, 0.95).passed is True
```

- [ ] **Step 2: Verify RED, then implement same-image one-to-one IoU matching**

Reject duplicate matches, unknown images, missing candidates, coordinate drift above one pixel, or score drift above 0.01. The CLI writes `parity-report.json` atomically.

- [ ] **Step 3: Run parity on frozen development validation inputs**

Use the same image order, confidence threshold, NMS IoU, letterbox color, and class-agnostic setting as Android. If a runtime lacks a verified desktop predictor, report `predictor_unavailable`; file conversion alone never counts as parity.

- [ ] **Step 4: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest ml\tests\test_mobile_benchmark.py -q
.\.venv\Scripts\python.exe -m pytest ml\tests -q
git add -- ml/src/crrc_vision/mobile_benchmark.py ml/tests/test_mobile_benchmark.py ml/scripts/evaluate_mobile_parity.py
git commit -m "feat(ml): evaluate mobile runtime parity"
```

### Task 6: Record the Phase 1 decision

**Files:**
- Create: `docs/validation/2026-09-01-mobile-runtime-export-phase1.md`
- Modify: `PROJECT_STATUS.md`

- [ ] **Step 1: Run fresh verification**

```powershell
.\.venv\Scripts\python.exe -m pytest ml\tests -q
.\gradlew.bat testDebugUnitTest assembleDebug
git diff --check
```

- [ ] **Step 2: Recompute protected hashes**

Record source model, converted artifacts, predictions, reports, and formal truth. Formal truth must remain `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`.

- [ ] **Step 3: Write fail-closed outcomes and commit**

Each runtime receives exactly one status: `parity_passed`, `conversion_failed`, `predictor_unavailable`, or `parity_failed`. Only `parity_passed` enters Phase 2.

```powershell
git add -- docs/validation/2026-09-01-mobile-runtime-export-phase1.md PROJECT_STATUS.md
git commit -m "docs: record mobile runtime export gate"
git status --short --branch
```

## Follow-on plans

1. Phase 2 integrates Phase 1 survivors behind the shared Android detector interface and runs P20 Pro cold, steady, and ten-minute thermal benchmarks.
2. Phase 3 trains NanoDet-Plus-m-1.5x 416 on the marked-point split, evaluates stock stride and a gated P2 variant, then benchmarks the survivor with ncnn.
3. Phase 4 adapts the existing PicoDet pipeline to the marked-point split, evaluates stock and gated P2 variants, and benchmarks Paddle Lite versus ncnn.
4. Phase 5 compares all survivors with the recall-first gate and packages only the winner.
