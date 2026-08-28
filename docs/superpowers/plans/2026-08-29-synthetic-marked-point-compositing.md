# Synthetic Marked-Point Compositing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an auditable, deterministic pipeline that turns real marked-fastener references into three-state local samples and composites approved samples into train-only real full images with transformed labels.

**Architecture:** Keep contracts, transforms, compositing, and auditing in focused Python modules under `ml/src/crrc_vision`. ImageGen outputs remain external and enter through a manifest-based ingest step; OpenCV applies bounded seeded transforms and label geometry. Every export is rejected unless its lineage is train-only, its labels validate, and formal truth retains the frozen hash.

**Tech Stack:** Python 3.11, dataclasses, Pillow, NumPy, OpenCV, pytest, COCO JSON, ImageGen external assets.

---

### Task 1: Synthetic data contracts and path/hash safety

**Files:**
- Create: `ml/src/crrc_vision/synthetic_contract.py`
- Test: `ml/tests/test_synthetic_contract.py`

- [ ] **Step 1: Write failing contract tests**

```python
from pathlib import Path
import pytest
from crrc_vision.synthetic_contract import (
    FROZEN_FORMAL_TRUTH_SHA256,
    SyntheticRecord,
    assert_external_output,
    assert_formal_truth_unchanged,
)

def test_synthetic_record_is_train_only():
    record = SyntheticRecord(
        sample_id="ref-01-normal-00",
        source_reference_sha256="a" * 64,
        source_scene_id="scene-01",
        state="NORMAL",
        image_path="locals/ref-01-normal-00.png",
    )
    assert record.synthetic is True
    assert record.eligible_split == "train"

def test_external_output_rejects_repo_child(tmp_path: Path):
    with pytest.raises(ValueError, match="Git外"):
        assert_external_output(tmp_path / "repo" / "out", tmp_path / "repo")

def test_formal_truth_hash_matches(tmp_path: Path):
    truth = tmp_path / "instances.json"
    truth.write_bytes(b"frozen")
    with pytest.raises(RuntimeError, match="formal truth"):
        assert_formal_truth_unchanged(truth, FROZEN_FORMAL_TRUTH_SHA256)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `./.venv/Scripts/python.exe -m pytest ml/tests/test_synthetic_contract.py -q`
Expected: collection fails because `crrc_vision.synthetic_contract` does not exist.

- [ ] **Step 3: Implement immutable records, validation, SHA-256, and path containment checks**

Implement `SyntheticRecord` with fixed `synthetic=True` and `eligible_split="train"`; reject any other split, unknown state, malformed hash, missing source scene, or absolute image path. Add `sha256_file`, `assert_formal_truth_unchanged`, and resolved-path `assert_external_output` that rejects output inside the repository.

- [ ] **Step 4: Run focused tests**

Run: `./.venv/Scripts/python.exe -m pytest ml/tests/test_synthetic_contract.py -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add ml/src/crrc_vision/synthetic_contract.py ml/tests/test_synthetic_contract.py
git commit -m "feat(vision): add synthetic data safety contract"
```

### Task 2: Seeded transforms with label propagation

**Files:**
- Create: `ml/src/crrc_vision/synthetic_transform.py`
- Test: `ml/tests/test_synthetic_transform.py`

- [ ] **Step 1: Write failing geometry tests**

```python
import numpy as np
from crrc_vision.synthetic_transform import TransformLimits, apply_homography_points, sample_transform

def test_sampled_transform_stays_inside_limits():
    limits = TransformLimits(rotation_deg=8, scale_min=0.85, scale_max=1.15, perspective_fraction=0.04)
    first = sample_transform(640, 480, seed=20260829, limits=limits)
    second = sample_transform(640, 480, seed=20260829, limits=limits)
    np.testing.assert_allclose(first.matrix, second.matrix)
    assert abs(first.rotation_deg) <= 8
    assert 0.85 <= first.scale <= 1.15

def test_point_round_trip_error_is_below_two_pixels():
    transform = sample_transform(640, 480, seed=7, limits=TransformLimits())
    points = np.array([[100.0, 80.0], [240.0, 200.0]], dtype=np.float32)
    warped = apply_homography_points(points, transform.matrix)
    restored = apply_homography_points(warped, np.linalg.inv(transform.matrix))
    assert np.max(np.linalg.norm(restored - points, axis=1)) < 2.0
```

- [ ] **Step 2: Run tests and verify failure**

Run: `./.venv/Scripts/python.exe -m pytest ml/tests/test_synthetic_transform.py -q`
Expected: missing module failure.

- [ ] **Step 3: Implement bounded deterministic transforms**

Implement `TransformLimits`, `SampledTransform`, `sample_transform`, `apply_homography_points`, `transform_bbox`, `warp_image_and_mask`, and `apply_photometric`. Use `numpy.random.default_rng(seed)`. Enforce rotation `±8°`, scale `0.85–1.15`, perspective displacement up to `4%`, brightness `0.8–1.2`, contrast `0.85–1.15`, JPEG `75–95`, noise sigma up to `4`, and blur kernel in `{1,3}`.

- [ ] **Step 4: Run focused tests**

Run: `./.venv/Scripts/python.exe -m pytest ml/tests/test_synthetic_transform.py -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add ml/src/crrc_vision/synthetic_transform.py ml/tests/test_synthetic_transform.py
git commit -m "feat(vision): add bounded synthetic transforms"
```

### Task 3: State geometry and audit

**Files:**
- Create: `ml/src/crrc_vision/synthetic_state.py`
- Test: `ml/tests/test_synthetic_state.py`

- [ ] **Step 1: Write failing state tests**

```python
from crrc_vision.synthetic_state import classify_state, relative_angle_deg, validate_state

def test_state_bands_are_separated():
    assert classify_state(2.0) == "NORMAL"
    assert classify_state(9.0) == "SLIGHT_LOOSE"
    assert classify_state(24.0) == "OBVIOUS_LOOSE"
    assert classify_state(4.5) == "UNCERTAIN"

def test_declared_state_must_match_endpoints():
    fixed = ((10.0, 10.0), (30.0, 10.0))
    moving = ((30.0, 10.0), (50.0, 10.0))
    assert relative_angle_deg(fixed, moving) == 0.0
    assert validate_state("NORMAL", fixed, moving).accepted is True
    assert validate_state("OBVIOUS_LOOSE", fixed, moving).accepted is False
```

- [ ] **Step 2: Run tests and verify failure**

Run: `./.venv/Scripts/python.exe -m pytest ml/tests/test_synthetic_state.py -q`
Expected: missing module failure.

- [ ] **Step 3: Implement endpoint-derived state validation**

Implement angle normalization, offset measurement, configurable bands (`0–3`, `6–12`, `18–35` degrees), rejection of degenerate line segments, and an audit result containing measured geometry and reason. Values between bands must be `UNCERTAIN`, never coerced.

- [ ] **Step 4: Run focused tests**

Run: `./.venv/Scripts/python.exe -m pytest ml/tests/test_synthetic_state.py -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add ml/src/crrc_vision/synthetic_state.py ml/tests/test_synthetic_state.py
git commit -m "feat(vision): validate synthetic looseness geometry"
```

### Task 4: Full-image compositor

**Files:**
- Create: `ml/src/crrc_vision/synthetic_composite.py`
- Test: `ml/tests/test_synthetic_composite.py`

- [ ] **Step 1: Write failing compositor tests**

```python
import numpy as np
import pytest
from crrc_vision.synthetic_composite import Placement, composite_sample

def test_composite_transforms_bbox_and_endpoints():
    background = np.zeros((240, 320, 3), dtype=np.uint8)
    patch = np.full((80, 100, 3), 140, dtype=np.uint8)
    mask = np.full((80, 100), 255, dtype=np.uint8)
    result = composite_sample(background, patch, mask, ((10, 10, 90, 70)),
                              (((20, 40), (45, 40)), ((45, 40), (75, 45))),
                              Placement(x=100, y=80, scale=1.0, rotation_deg=0.0))
    assert result.bbox_xyxy == pytest.approx((110, 90, 190, 150), abs=1.0)
    assert result.image.shape == background.shape

def test_composite_rejects_border_collision():
    background = np.zeros((100, 100, 3), dtype=np.uint8)
    patch = np.zeros((80, 80, 3), dtype=np.uint8)
    mask = np.full((80, 80), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="边界"):
        composite_sample(background, patch, mask, ((0, 0, 80, 80)),
                         (((10, 10), (20, 10)), ((20, 10), (30, 15))),
                         Placement(x=60, y=60, scale=1.0, rotation_deg=0.0))
```

- [ ] **Step 2: Run tests and verify failure**

Run: `./.venv/Scripts/python.exe -m pytest ml/tests/test_synthetic_composite.py -q`
Expected: missing module failure.

- [ ] **Step 3: Implement mask blending and label projection**

Implement seeded placement, homography projection, soft-mask alpha blending with optional OpenCV seamless clone, boundary checks, minimum 12-pixel target short side, maximum three inserted points, and overlap rejection against supplied existing boxes. Return the transformed bbox, endpoints, anchor, homography, and seam metrics.

- [ ] **Step 4: Run focused tests**

Run: `./.venv/Scripts/python.exe -m pytest ml/tests/test_synthetic_composite.py -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add ml/src/crrc_vision/synthetic_composite.py ml/tests/test_synthetic_composite.py
git commit -m "feat(vision): composite marked points into full images"
```

### Task 5: Manifest audit and train-only export

**Files:**
- Create: `ml/src/crrc_vision/synthetic_audit.py`
- Create: `ml/scripts/build_synthetic_reference_pack.py`
- Create: `ml/scripts/audit_synthetic_dataset.py`
- Test: `ml/tests/test_synthetic_audit.py`

- [ ] **Step 1: Write failing audit tests**

```python
from crrc_vision.synthetic_audit import audit_records

def test_audit_rejects_validation_lineage():
    result = audit_records([{
        "sample_id": "x", "synthetic": True, "eligible_split": "train",
        "source_split": "val", "source_reference_sha256": "a" * 64,
        "state": "NORMAL", "review_status": "APPROVED",
    }])
    assert result.passed is False
    assert "source_split" in result.errors[0]

def test_audit_requires_balanced_approved_states():
    records = [{
        "sample_id": f"n-{i}", "synthetic": True, "eligible_split": "train",
        "source_split": "train", "source_reference_sha256": f"{i:064x}",
        "state": "NORMAL", "review_status": "APPROVED",
    } for i in range(8)]
    result = audit_records(records)
    assert result.passed is False
    assert result.approved_by_state["NORMAL"] == 8
```

- [ ] **Step 2: Run tests and verify failure**

Run: `./.venv/Scripts/python.exe -m pytest ml/tests/test_synthetic_audit.py -q`
Expected: missing module failure.

- [ ] **Step 3: Implement reference-pack builder and dataset audit**

The reference-pack command must read the existing reviewed marked-point COCO, choose at most one point per scene, reject low-quality or non-train sources, create 12 context crops, and write `references.json` plus one ImageGen prompt per reference. The audit command must validate hashes, dimensions, state balance, review status, pHash leakage, label bounds, source grouping, and formal truth hash; it exits non-zero on any violation.

- [ ] **Step 4: Run focused tests and CLI help**

Run: `./.venv/Scripts/python.exe -m pytest ml/tests/test_synthetic_audit.py -q`
Expected: all tests pass.

Run: `./.venv/Scripts/python.exe ml/scripts/build_synthetic_reference_pack.py --help`
Expected: exit code 0 and options for data root, reviewed COCO, output, and count.

- [ ] **Step 5: Commit**

```powershell
git add ml/src/crrc_vision/synthetic_audit.py ml/scripts/build_synthetic_reference_pack.py ml/scripts/audit_synthetic_dataset.py ml/tests/test_synthetic_audit.py
git commit -m "feat(vision): audit synthetic marked-point lineage"
```

### Task 6: ImageGen ingest and deterministic full-image build

**Files:**
- Create: `ml/scripts/ingest_synthetic_locals.py`
- Create: `ml/scripts/build_synthetic_full_images.py`
- Test: `ml/tests/test_synthetic_pipeline.py`

- [ ] **Step 1: Write failing pipeline tests**

```python
from pathlib import Path
import pytest
from crrc_vision.synthetic_contract import FROZEN_FORMAL_TRUTH_SHA256

def test_ingest_rejects_missing_sidecar(run_ingest, tmp_path: Path):
    image = tmp_path / "ref-01-normal.png"
    image.write_bytes(b"not-an-image")
    with pytest.raises(RuntimeError, match="sidecar"):
        run_ingest([image], output=tmp_path / "external")

def test_build_is_reproducible(run_full_build, approved_fixture):
    first = run_full_build(approved_fixture, seed=20260829)
    second = run_full_build(approved_fixture, seed=20260829)
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.formal_truth_sha256 == FROZEN_FORMAL_TRUTH_SHA256
```

- [ ] **Step 2: Run tests and verify failure**

Run: `./.venv/Scripts/python.exe -m pytest ml/tests/test_synthetic_pipeline.py -q`
Expected: fixture/module behavior is not implemented.

- [ ] **Step 3: Implement ingest and build commands**

Ingest must require one JSON sidecar per local image containing bbox, two line segments, anchor, declared state, prompt hash, reference hash, and review state. The full-image builder must use only approved train records/backgrounds, derive per-sample seeds from the global seed and sample ID, cap inserts at three, write images atomically, and export manifest plus COCO boxes and endpoint annotations.

- [ ] **Step 4: Run focused tests**

Run: `./.venv/Scripts/python.exe -m pytest ml/tests/test_synthetic_pipeline.py -q`
Expected: all tests pass and two identical seeded runs have identical manifests.

- [ ] **Step 5: Commit**

```powershell
git add ml/scripts/ingest_synthetic_locals.py ml/scripts/build_synthetic_full_images.py ml/tests/test_synthetic_pipeline.py
git commit -m "feat(vision): build reproducible synthetic full images"
```

### Task 7: Run the 12-reference pilot and document evidence

**Files:**
- Create: `docs/validation/2026-08-29-synthetic-marked-point-pilot.md`
- Modify: `README.md`
- Modify: `PROJECT_STATUS.md`
- External only: `E:/Work/京新数智/识动hicool/中车眼镜数据资产/synthetic/marked-point-v1/**`

- [ ] **Step 1: Verify baseline and frozen truth**

Run: `Get-FileHash E:/crrc_vision_data/annotations/fastener-v2/instances.json -Algorithm SHA256`
Expected: `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`.

- [ ] **Step 2: Build and visually inspect the reference pack**

Run: `./.venv/Scripts/python.exe ml/scripts/build_synthetic_reference_pack.py --data-root E:/crrc_vision_data --count 12 --output E:/crrc_vision_data/synthetic/marked-point-v1/reference-pack`
Expected: 12 unique train-scene references, 12 context crops, 12 prompts, and no val/sealed lineage.

- [ ] **Step 3: Generate 36 local candidates from the 12 references**

Use ImageGen once per asset with the corresponding local reference image. Generate `NORMAL`, `SLIGHT_LOOSE`, and `OBVIOUS_LOOSE` for each reference, preserving the real industrial camera style. Store outputs and generation metadata only under the external pilot directory.

- [ ] **Step 4: Annotate and audit every local candidate**

Record bbox, fixed/moving endpoints, anchor, state, visibility, prompt hash, reference hash, and independent review result. Run the audit and require at least eight approved samples per state, at least 75% overall approval, zero bounds errors, and zero lineage violations.

- [ ] **Step 5: Build and audit full-image composites**

Run: `./.venv/Scripts/python.exe ml/scripts/build_synthetic_full_images.py --input E:/crrc_vision_data/synthetic/marked-point-v1/approved-locals.json --output E:/crrc_vision_data/synthetic/marked-point-v1/full-images --seed 20260829`
Expected: at least 24 approved full images, at least six per state, no more than three inserts per image, and deterministic manifest hash.

- [ ] **Step 6: Run the complete regression and safety gate**

Run: `./.venv/Scripts/python.exe -m pytest ml/tests -q`
Expected: all tests pass.

Run: `./.venv/Scripts/python.exe ml/scripts/audit_synthetic_dataset.py --root E:/crrc_vision_data/synthetic/marked-point-v1`
Expected: exit code 0 with state counts, rejection buckets, lineage checks, and unchanged formal truth hash.

- [ ] **Step 7: Write evidence and update project handoff**

Document actual generated/approved/rejected counts, state distribution, manifest hashes, formal truth hash, limitations, and whether the pilot is eligible for a synthetic-training ablation. Explicitly state that real-val accuracy and end-to-end looseness accuracy remain unproven.

- [ ] **Step 8: Commit repository-only artifacts**

```powershell
git add README.md PROJECT_STATUS.md docs/validation/2026-08-29-synthetic-marked-point-pilot.md
git commit -m "docs(vision): record synthetic marked-point pilot"
```
