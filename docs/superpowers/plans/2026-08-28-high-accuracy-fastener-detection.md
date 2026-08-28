# High-Accuracy Full-Image Fastener Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a scene-isolated, auditable real-data loop that raises full-image physical-target detection to sealed-test recall at least 95%, precision at least 90%, and complete-scene rate at least 90% without tuning on the sealed test.

**Architecture:** Freeze deterministic train/validation/sealed-test scene partitions before labeling, extend the existing two-pass Codex review pipeline to all 97 previously unused scene groups, and train a stride-4 P2 mobile student across three fixed seeds. Model selection and thresholds use validation only; a one-shot sealed evaluator enforces the production accuracy gate and permanently records when the seal is opened.

**Tech Stack:** Python 3.11+, pytest, Pillow/OpenCV, COCO JSON, PyTorch 2.7, Ultralytics 8.2.40 internal challenger, ONNX, existing Git-external CRRC vision asset store.

---

## File map

- `ml/src/crrc_vision/high_accuracy_split.py`: deterministic partitioning, split integrity, and seal manifests.
- `ml/scripts/build_high_accuracy_split.py`: Git-external selection CLI using the 482-image manifest and current reviewed COCO.
- `ml/src/crrc_vision/high_accuracy_gate.py`: threshold selection, IoU matching, scene completeness, multi-seed stability, and one-shot test gate.
- `ml/scripts/evaluate_high_accuracy.py`: validation and sealed-test evaluation CLI.
- `ml/src/crrc_vision/p2_training.py`: secure multi-seed training command/manifests and conservative augmentation contract.
- `ml/scripts/train_p2_high_accuracy.py`: guarded external-runtime training entry point.
- `ml/src/crrc_vision/error_buckets.py`: deterministic FP/FN error taxonomy and review selection.
- `ml/scripts/build_error_review_pack.py`: Git-external overlays and error-bucket audit pack.
- Existing `ml/scripts/build_codex_review_pack.py`, `ml/scripts/assemble_reviewed_coco.py`, and `ml/src/crrc_vision/reviewed_coco.py`: reused for two-pass full-image review and cumulative reviewed COCO assembly.
- Tests mirror each module under `ml/tests/`.

### Task 1: Freeze deterministic scene partitions

**Files:**
- Create: `ml/tests/test_high_accuracy_split.py`
- Create: `ml/src/crrc_vision/high_accuracy_split.py`
- Create: `ml/scripts/build_high_accuracy_split.py`

- [ ] **Step 1: Write failing quota and isolation tests**

```python
def test_partition_uses_all_177_scenes_without_overlap():
    result = build_high_accuracy_partition(
        manifest_rows=rows_for_177_scenes(),
        existing_train_scenes=set(scene_ids(1, 64)),
        existing_val_scenes=set(scene_ids(65, 80)),
        new_train_count=52,
        new_val_count=15,
        sealed_test_count=30,
        seed=20260828,
    )
    assert len(result.train_scenes) == 116
    assert len(result.val_scenes) == 31
    assert len(result.sealed_test_scenes) == 30
    assert not (set(result.train_scenes) & set(result.val_scenes))
    assert not (set(result.train_scenes) & set(result.sealed_test_scenes))
    assert not (set(result.val_scenes) & set(result.sealed_test_scenes))
```

Add tests that reject a scene/hash/path/image ID appearing in two partitions, reject any previously trained scene in sealed-test, and produce byte-identical JSON from reordered input rows.

- [ ] **Step 2: Run the tests and verify they fail**

Run: `.venv\Scripts\python.exe -m pytest ml\tests\test_high_accuracy_split.py -q`

Expected: import failure for `crrc_vision.high_accuracy_split`.

- [ ] **Step 3: Implement the partition contract**

```python
@dataclass(frozen=True)
class HighAccuracyPartition:
    train_scenes: tuple[str, ...]
    val_scenes: tuple[str, ...]
    sealed_test_scenes: tuple[str, ...]
    scene_representatives: dict[str, str]
    seed: int


def assert_partition_isolated(document: dict[str, object]) -> None:
    owners: dict[tuple[str, object], str] = {}
    for partition in ("train", "val", "sealed_test"):
        for row in document[partition]:
            for field in ("scene_group", "sha256", "image_id", "relative_path"):
                key = (field, row[field])
                previous = owners.setdefault(key, partition)
                if previous != partition:
                    raise ValueError(f"HIGH_ACCURACY_SPLIT_LEAKAGE:{field}")
```

For each unused scene, select one representative frame by descending focus score, then deterministic path. Stratify unused scenes by original split, capture-time quartile, focus quartile, grayscale-brightness quartile, and current fused-candidate-count quartile. Within each stratum, order by `sha256(seed + scene_group)` and allocate sealed-test first, new validation second, and new training last until exact 30/15/52 quotas are met.

- [ ] **Step 4: Implement the split CLI**

The CLI reads `manifest.jsonl`, `annotations/silver-gate-cumulative-013/instances.silver.json`, and `runs/safe-auto-candidates-v2.2/candidates.json`; computes brightness from original images; verifies the formal-truth hash; and writes only below `selections/high-accuracy-v1/`:

```json
{
  "schema_version": "high-accuracy-partition-v1",
  "seed": 20260828,
  "sealed_test_opened": false,
  "train": [],
  "val": [],
  "sealed_test": [],
  "input_hashes": {}
}
```

Require an empty output directory and use atomic JSON replacement.

- [ ] **Step 5: Run focused and full tests**

Run: `.venv\Scripts\python.exe -m pytest ml\tests\test_high_accuracy_split.py ml\tests -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add ml/tests/test_high_accuracy_split.py ml/src/crrc_vision/high_accuracy_split.py ml/scripts/build_high_accuracy_split.py
git commit -m "feat(vision): freeze high-accuracy scene partitions"
```

### Task 2: Build review selections and high-resolution packs

**Files:**
- Modify: `ml/src/crrc_vision/codex_review_pack.py`
- Modify: `ml/scripts/build_codex_review_pack.py`
- Modify: `ml/tests/test_codex_review.py`
- Create: `ml/tests/test_codex_review_pack.py`

- [ ] **Step 1: Add failing partition-aware pack tests**

```python
def test_pack_copies_partition_and_never_exposes_sealed_labels(tmp_path):
    summary = build_pack(
        candidates,
        source_root,
        tmp_path,
        selected_relative_paths=["a.jpg"],
        partition="sealed_test",
        include_existing_decisions=False,
    )
    assert summary.images == 1
    assert not list(tmp_path.rglob("*decision*"))
```

Also assert four overlapping high-resolution scan tiles cover every source pixel, candidate IDs are exhaustive, and a non-empty output is refused.

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv\Scripts\python.exe -m pytest ml\tests\test_codex_review.py ml\tests\test_codex_review_pack.py -q`

Expected: failure because `partition` is not accepted.

- [ ] **Step 3: Add partition metadata without changing review semantics**

Store partition name, partition-manifest SHA-256, candidates SHA-256, source-image hashes, and formal-truth before/after hashes in `integrity.json`. Sealed-test packs may contain images and unlabeled candidate geometry for blind annotation, but may not include model evaluation, previous decisions, scores in reviewer-visible sheets, or train/validation labels.

- [ ] **Step 4: Generate three Git-external packs**

Run the CLI separately for new train, new validation, and sealed-test representatives using `safe-auto-candidates-v2.2/candidates.json`. Write to `review-packs/high-accuracy-v1/{train,val,sealed-test}` and verify 52/15/30 independent scene groups.

- [ ] **Step 5: Run full tests and commit**

```powershell
.venv\Scripts\python.exe -m pytest ml\tests -q
git add ml/src/crrc_vision/codex_review_pack.py ml/scripts/build_codex_review_pack.py ml/tests/test_codex_review.py ml/tests/test_codex_review_pack.py
git commit -m "feat(vision): build partition-aware review packs"
```

### Task 3: Complete two-pass full-image review and assemble high-accuracy COCO

**Files:**
- Create: `ml/tests/test_high_accuracy_dataset.py`
- Create: `ml/src/crrc_vision/high_accuracy_dataset.py`
- Create: `ml/scripts/assemble_high_accuracy_dataset.py`
- Reuse: `ml/src/crrc_vision/reviewed_coco.py`

- [ ] **Step 1: Add failing assembly tests**

Test that accepted `fastener` and `pipe_joint` boxes both become category 1 `fastener_target`, uncertain images never enter complete datasets, train/val/sealed-test scene groups remain disjoint, and the original 80 reviewed scenes merge only into their frozen train/validation partitions.

```python
assert result["categories"] == [{"id": 1, "name": "fastener_target"}]
assert result["counts"] == {"train_scenes": 116, "val_scenes": 31, "test_scenes": 30}
```

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv\Scripts\python.exe -m pytest ml\tests\test_high_accuracy_dataset.py -q`

Expected: import failure.

- [ ] **Step 3: Implement guarded assembly**

The assembler consumes only contract-valid first/second-pass documents, verifies every candidate ID and source hash, merges the current 80-scene reviewed COCO, maps both source categories to one physical-target category, and writes separate train/val/sealed-test COCO plus a manifest. It refuses duplicate scenes, empty validation/test, fewer than 30 complete sealed-test scenes, fewer than 200 sealed-test boxes, any synthetic test image, and any formal-truth hash change.

- [ ] **Step 4: Perform Codex first and blind second review**

For every selected image: inspect full-resolution whole image and four overlap scans; decide all candidates; add independently bounded misses; keep ambiguity as uncertain; and hide first-pass decisions from geometric second review. Complete each partition before assembly; never report a partial partition as complete.

- [ ] **Step 5: Assemble and verify**

Write Git-external outputs under `annotations/high-accuracy-v1/`. Record complete scenes, accepted boxes, uncertain images, class counts before merge, split counts, source hashes, and formal-truth SHA-256.

- [ ] **Step 6: Run tests and commit**

```powershell
.venv\Scripts\python.exe -m pytest ml\tests -q
git add ml/tests/test_high_accuracy_dataset.py ml/src/crrc_vision/high_accuracy_dataset.py ml/scripts/assemble_high_accuracy_dataset.py
git commit -m "feat(vision): assemble isolated high-accuracy dataset"
```

### Task 4: Implement validation threshold and sealed-test gates

**Files:**
- Create: `ml/tests/test_high_accuracy_gate.py`
- Create: `ml/src/crrc_vision/high_accuracy_gate.py`
- Create: `ml/scripts/evaluate_high_accuracy.py`

- [ ] **Step 1: Write failing metric tests**

```python
def test_gate_selects_threshold_on_val_and_reuses_it_on_test():
    threshold = select_threshold(val_predictions, val_truth, minimum_precision=0.90)
    report = evaluate_at_threshold(test_predictions, test_truth, threshold=threshold)
    assert report.threshold == threshold


def test_complete_scene_rate_requires_every_target_match():
    report = evaluate_at_threshold(predictions_missing_one_box, truth, threshold=0.2)
    assert report.complete_scenes == 0
```

Add exact cases for duplicate predictions, one-to-one IoU matching, empty scenes, boundary boxes, score ties, `recall_at_precision_90`, three-seed mean/std/range, and gate failure when test has fewer than 30 scenes or 200 boxes.

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv\Scripts\python.exe -m pytest ml\tests\test_high_accuracy_gate.py -q`

Expected: import failure.

- [ ] **Step 3: Implement pure metrics**

```python
@dataclass(frozen=True)
class AccuracyReport:
    precision: float
    recall: float
    complete_scene_rate: float
    threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    passed: bool
```

Use deterministic greedy score ordering with one-to-one IoU 0.50 matching. Select the highest-recall validation threshold whose precision is at least 0.90; tie-break toward higher precision, then higher threshold. Freeze this value in `selection-manifest.json` before test evaluation.

- [ ] **Step 4: Implement one-shot seal handling**

The CLI has separate `--mode val` and `--mode sealed-test`. Validation cannot read the sealed-test COCO path. Sealed-test requires the frozen validation manifest, refuses when `sealed_test_opened` is already true, atomically writes an opened audit record before loading predictions, and records the exact model/export/prediction/test hashes whether evaluation passes or fails.

- [ ] **Step 5: Run tests and commit**

```powershell
.venv\Scripts\python.exe -m pytest ml\tests -q
git add ml/tests/test_high_accuracy_gate.py ml/src/crrc_vision/high_accuracy_gate.py ml/scripts/evaluate_high_accuracy.py
git commit -m "feat(vision): enforce validation and sealed accuracy gates"
```

### Task 5: Add secure three-seed P2 training manifests

**Files:**
- Create: `ml/tests/test_p2_training.py`
- Create: `ml/src/crrc_vision/p2_training.py`
- Create: `ml/scripts/train_p2_high_accuracy.py`
- Modify: `ml/src/crrc_vision/yolo_p2.py`

- [ ] **Step 1: Write failing command and integrity tests**

Assert seeds are exactly `20260828`, `20260829`, and `20260830`; training cannot see sealed-test paths; output directories must be empty; every source hash is checked; Ultralytics must be 8.2.40; checkpoints use `weights_only=True` plus the existing global allowlist; and generated YAML contains only train/val.

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv\Scripts\python.exe -m pytest ml\tests\test_p2_training.py -q`

Expected: import failure.

- [ ] **Step 3: Implement conservative training configurations**

Use full images plus deterministic 2x2 12%-overlap training tiles, preserve negative tiles, and cap each scene's sampled views per epoch. Fix P2-S input to 640 for the baseline. Use bounded brightness/contrast, light blur/noise, rotation within 5 degrees, and scale 0.8-1.2; disable perspective, copy-paste, and aggressive mosaic. Early stopping observes validation only with patience 15, while retaining best and last checkpoints.

- [ ] **Step 4: Implement run manifests**

Each run records code commit, runtime versions, seed, dataset/partition/formal-truth hashes, pretrained checkpoint hash, augmentation values, image size, batch size, learning rate, epoch, selected checkpoint, and license status. Reject any path that resolves outside the Git-external asset root.

- [ ] **Step 5: Run tests and commit**

```powershell
.venv\Scripts\python.exe -m pytest ml\tests -q
git add ml/tests/test_p2_training.py ml/src/crrc_vision/p2_training.py ml/scripts/train_p2_high_accuracy.py ml/src/crrc_vision/yolo_p2.py
git commit -m "feat(vision): add reproducible three-seed P2 training"
```

### Task 6: Train the baseline and one bounded precision challenger

**Files:**
- Git-external: `runs/high-accuracy-p2-s-640-seed-{20260828,20260829,20260830}/`
- Git-external: `runs/high-accuracy-p2-challenger/`
- Modify: `PROJECT_STATUS.md`

- [ ] **Step 1: Train P2-S 640 across three seeds**

Run the guarded CLI in the pinned external runtime. Evaluate every best checkpoint on validation with full-image class-agnostic NMS. Record mean, standard deviation, worst seed, per-size recall, complete-scene rate, false positives per image, latency, and model size.

- [ ] **Step 2: Apply stability gates**

Reject the configuration if recall range exceeds 0.05, any run has train/val scene leakage, validation precision cannot reach 0.90, or two consecutive data rounds improve recall by less than 0.01.

- [ ] **Step 3: Train one larger P2 challenger**

Train exactly one YOLOv8m-P2 640 stride-4 challenger on the same dataset, seeds, augmentations, and validation rules. It is an internal accuracy ceiling, not a mobile default. Continue to distillation only when its mean validation recall at precision 0.90 exceeds P2-S by at least 0.03.

- [ ] **Step 4: Freeze the winner before test**

Select the configuration by highest three-seed mean validation `recall_at_precision_90`; tie-break by worst-seed recall, model size, then full-image latency. Freeze checkpoint, input size, NMS, threshold, and prediction hash in the selection manifest.

- [ ] **Step 5: Update durable status and commit**

Document all validation results, including failures and variance, without sealed-test metrics.

```powershell
git add PROJECT_STATUS.md
git commit -m "docs(vision): record high-accuracy validation candidates"
```

### Task 7: Build deterministic FP/FN error buckets

**Files:**
- Create: `ml/tests/test_error_buckets.py`
- Create: `ml/src/crrc_vision/error_buckets.py`
- Create: `ml/scripts/build_error_review_pack.py`

- [ ] **Step 1: Write failing taxonomy tests**

Use image brightness, Laplacian focus, target area, border distance, occlusion flags, local contrast, and nearby-object density to assign each validation error to exactly one primary bucket and zero or more secondary tags. Score ties use a fixed priority: annotation dispute, border truncation, tiny, dark, blur, occlusion, reflection, dense pipes, lookalike.

- [ ] **Step 2: Implement and test**

Create whole-image overlays and crops for validation only. Each record includes image/scene/prediction/ground-truth IDs, error kind, bucket, evidence values, and source hashes. The builder refuses sealed-test inputs by partition field and hash.

- [ ] **Step 3: Use at most three data rounds**

Round 2 and Round 3 may add train examples or bounded augmentation only for dominant validation buckets. Do not change the sealed-test partition or inspect sealed-test predictions. Re-run all three seeds after each material dataset change.

- [ ] **Step 4: Commit**

```powershell
.venv\Scripts\python.exe -m pytest ml\tests -q
git add ml/tests/test_error_buckets.py ml/src/crrc_vision/error_buckets.py ml/scripts/build_error_review_pack.py
git commit -m "feat(vision): classify validation detection errors"
```

### Task 8: Open the sealed test exactly once

**Files:**
- Git-external: `runs/high-accuracy-sealed-test-v1/`
- Create: `docs/validation/2026-08-28-high-accuracy-sealed-test.md`
- Modify: `PROJECT_STATUS.md`

- [ ] **Step 1: Verify preconditions**

Require clean Git status, passing full Python suite, formal-truth hash unchanged, 30 complete test scenes, at least 200 test boxes, frozen model and threshold manifest, three-seed validation report, no synthetic test images, and `sealed_test_opened=false`.

- [ ] **Step 2: Run predictions and one-shot gate**

Generate predictions without reading test annotations, then invoke `evaluate_high_accuracy.py --mode sealed-test`. The audit marker must be written even if inference or evaluation fails so the same test cannot be silently reused.

- [ ] **Step 3: Apply the hard decision**

PASS only when recall is at least 0.95, precision at least 0.90, and complete-scene rate at least 0.90. Any failure keeps Android integration disabled and turns this test into a diagnostic set; the next production claim requires newly collected external scenes.

- [ ] **Step 4: Document and commit**

Record exact TP/FP/FN, per-scene misses, small-target recall, test/model/prediction hashes, threshold, runtime, and pass/fail. Never describe AP50 as direct accuracy.

```powershell
git add docs/validation/2026-08-28-high-accuracy-sealed-test.md PROJECT_STATUS.md
git commit -m "docs(vision): record one-shot sealed accuracy gate"
```

### Task 9: Export only a passing and licensable model

**Files:**
- Git-external: `exports/android/high-accuracy-v1/`
- Modify only after PASS: `app/src/main/java/com/ar/glass/vision/DefaultImageAnalyzer.java`
- Create only after PASS: `app/src/test/java/com/ar/glass/vision/OnnxFastenerAnalyzerTest.java`

- [ ] **Step 1: Enforce export refusal before PASS**

Add a test that export refuses a missing/failed sealed gate or unresolved license status. Internal challenger weights remain research-only.

- [ ] **Step 2: Export and verify ONNX after PASS**

Export fixed-shape ONNX, run `onnx.checker`, compare PyTorch/ONNX predictions on the validation set, and require maximum box-coordinate difference below 1 pixel and score difference below 0.01 after matched NMS.

- [ ] **Step 3: Integrate and benchmark Android**

Only after licensing is clear, implement resize/letterbox, ONNX inference, physical-target class merge, class-agnostic NMS, coordinate restoration, and ROI handoff. On the specified phone run 50 consecutive end-to-end inferences and record P50/P95, peak memory, temperature, and throttling.

- [ ] **Step 4: Final verification**

```powershell
.venv\Scripts\python.exe -m pytest ml\tests -q
.\gradlew.bat testDebugUnitTest assembleDebug
git diff --check
git status --short
```

Verify formal-truth SHA-256 remains `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001` and no image/model artifact is tracked by Git.
