# Physical Candidate Deduplication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace IoU-only single-link fusion with deterministic conservative physical-candidate deduplication and produce auditable v2 candidate statistics without changing formal truth.

**Architecture:** Keep `Candidate` and `FusedCandidate` as the stable boundary. Add a composite pair predicate and complete-link clustering inside `auto_labeling.py`, then make the candidate builder emit a new v2 artifact with immutable input/truth hashes and cluster-size metrics. Validate first with unit counterexamples, then with the external 0007/0047 scenes and a manual full-resolution entity-merge check.

**Tech Stack:** Python 3.11+, pytest, dataclasses, JSON, existing CRRC vision utilities.

---

### Task 1: Conservative geometric duplicate predicate

**Files:**
- Modify: `ml/tests/test_auto_labeling.py`
- Modify: `ml/src/crrc_vision/auto_labeling.py`

- [ ] **Step 1: Write the failing nested-box and adjacent-object tests**

```python
def test_nested_same_center_candidates_merge_below_iou_threshold() -> None:
    fused = fuse_candidates([
        candidate("small", "hsv", "fastener", (0, 0, 20, 20), 0.8),
        candidate("large", "reference_teacher", "fastener", (-5, -5, 25, 25), 0.9),
    ])
    assert len(fused) == 1
    assert len(fused[0].member_ids) == 2


def test_adjacent_objects_are_not_merged_by_containment_rule() -> None:
    fused = fuse_candidates([
        candidate("left", "hsv", "fastener", (0, 0, 20, 20), 0.8),
        candidate("spanning", "reference_teacher", "fastener", (0, 0, 40, 20), 0.9),
    ])
    assert len(fused) == 2
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/test_auto_labeling.py::test_nested_same_center_candidates_merge_below_iou_threshold tests/test_auto_labeling.py::test_adjacent_objects_are_not_merged_by_containment_rule -v
```

Expected: the nested-box test fails with `assert 2 == 1`; the adjacent-object test passes under the old implementation.

- [ ] **Step 3: Implement the minimal composite pair predicate**

Add named defaults and helpers in `ml/src/crrc_vision/auto_labeling.py`:

```python
DEFAULT_CONTAINMENT_THRESHOLD = 0.85
DEFAULT_CENTER_DISTANCE_THRESHOLD = 0.25
DEFAULT_MAX_AREA_RATIO = 4.0


def _box_area(box: Box) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def _intersection_area(left: Box, right: Box) -> float:
    return max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )


def _is_geometric_duplicate(left: Box, right: Box, iou_threshold: float) -> bool:
    if iou_xyxy(left, right) >= iou_threshold:
        return True
    left_area = _box_area(left)
    right_area = _box_area(right)
    smaller = left if left_area <= right_area else right
    smaller_area = min(left_area, right_area)
    if _intersection_area(left, right) / smaller_area < DEFAULT_CONTAINMENT_THRESHOLD:
        return False
    if max(left_area, right_area) / smaller_area > DEFAULT_MAX_AREA_RATIO:
        return False
    left_center = ((left[0] + left[2]) / 2, (left[1] + left[3]) / 2)
    right_center = ((right[0] + right[2]) / 2, (right[1] + right[3]) / 2)
    center_distance = (
        (left_center[0] - right_center[0]) ** 2
        + (left_center[1] - right_center[1]) ** 2
    ) ** 0.5
    smaller_diagonal = (
        (smaller[2] - smaller[0]) ** 2 + (smaller[3] - smaller[1]) ** 2
    ) ** 0.5
    return center_distance / smaller_diagonal <= DEFAULT_CENTER_DISTANCE_THRESHOLD
```

Update `iou_xyxy` to reuse `_intersection_area` and `_box_area`, and update clustering to call `_is_geometric_duplicate`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: 2 passed.

- [ ] **Step 5: Commit the first red-green slice**

```powershell
git add -- ml/tests/test_auto_labeling.py ml/src/crrc_vision/auto_labeling.py
git commit -m "feat: merge conservative nested candidates"
```

### Task 2: Complete-link, deterministic clustering

**Files:**
- Modify: `ml/tests/test_auto_labeling.py`
- Modify: `ml/src/crrc_vision/auto_labeling.py`

- [ ] **Step 1: Write failing chain-bridge and input-order tests**

```python
def test_complete_link_prevents_chain_bridge() -> None:
    fused = fuse_candidates([
        candidate("a", "hsv", "fastener", (0, 0, 20, 20), 0.8),
        candidate("b", "student", "fastener", (5, 0, 25, 20), 0.8),
        candidate("c", "reference_teacher", "fastener", (10, 0, 30, 20), 0.8),
    ])
    assert sorted(len(item.member_ids) for item in fused) == [1, 2]


def test_fusion_is_deterministic_across_input_order() -> None:
    rows = [
        candidate("a", "hsv", "fastener", (0, 0, 20, 20), 0.7),
        candidate("b", "student", "fastener", (5, 0, 25, 20), 0.8),
        candidate("c", "reference_teacher", "fastener", (10, 0, 30, 20), 0.9),
    ]
    forward = fuse_candidates(rows)
    backward = fuse_candidates(list(reversed(rows)))
    assert [item.stable_id() for item in forward] == [
        item.stable_id() for item in backward
    ]
```

- [ ] **Step 2: Run the chain-bridge test and verify RED**

Run:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/test_auto_labeling.py::test_complete_link_prevents_chain_bridge -v
```

Expected: FAIL because single-link returns one three-member cluster.

- [ ] **Step 3: Replace single-link clustering with complete-link clustering**

```python
def _complete_link_cluster(
    seed: Candidate,
    pending: list[Candidate],
    iou_threshold: float,
) -> tuple[list[Candidate], list[Candidate]]:
    cluster = [seed]
    remaining: list[Candidate] = []
    for row in pending:
        matches = row.relative_path == seed.relative_path and all(
            _is_geometric_duplicate(row.xyxy, member.xyxy, iou_threshold)
            for member in cluster
        )
        if matches:
            cluster.append(row)
        else:
            remaining.append(row)
    return cluster, remaining
```

Call `_complete_link_cluster` from `fuse_candidates`. Preserve the existing deterministic sort, weighted fusion, source-family rules, and conflict behavior.

- [ ] **Step 4: Run all auto-labeling tests and verify GREEN**

Run:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/test_auto_labeling.py -v
```

Expected: all tests pass, including the existing cross-category `conflict` test.

- [ ] **Step 5: Commit the complete-link slice**

```powershell
git add -- ml/tests/test_auto_labeling.py ml/src/crrc_vision/auto_labeling.py
git commit -m "fix: prevent candidate chain bridging"
```

### Task 3: Auditable v2 statistics and immutable output

**Files:**
- Modify: `ml/tests/test_auto_labeling.py`
- Modify: `ml/src/crrc_vision/auto_labeling.py`
- Modify: `ml/scripts/build_safe_auto_candidates.py`

- [ ] **Step 1: Write the failing fusion-statistics test**

```python
from crrc_vision.auto_labeling import fusion_stats


def test_fusion_stats_report_reduction_and_cluster_sizes() -> None:
    rows = [
        candidate("a", "hsv", "fastener", (0, 0, 20, 20), 0.8),
        candidate("b", "reference_teacher", "fastener", (1, 1, 21, 21), 0.9),
        candidate("c", "student", "fastener", (100, 100, 120, 120), 0.7),
    ]
    fused = fuse_candidates(rows)
    assert fusion_stats(rows, fused) == {
        "raw_candidates": 3,
        "fused_candidates": 2,
        "candidate_reduction": 1,
        "cluster_size_histogram": {"1": 1, "2": 1},
    }
```

- [ ] **Step 2: Run the statistics test and verify RED**

Run:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/test_auto_labeling.py::test_fusion_stats_report_reduction_and_cluster_sizes -v
```

Expected: collection error because `fusion_stats` is not defined.

- [ ] **Step 3: Implement statistics and wire the v2 manifest**

Add to `auto_labeling.py`:

```python
def fusion_stats(
    rows: list[Candidate], fused: list[FusedCandidate]
) -> dict[str, int | dict[str, int]]:
    histogram: dict[str, int] = {}
    for item in fused:
        key = str(len(item.member_ids))
        histogram[key] = histogram.get(key, 0) + 1
    return {
        "raw_candidates": len(rows),
        "fused_candidates": len(fused),
        "candidate_reduction": len(rows) - len(fused),
        "cluster_size_histogram": dict(sorted(histogram.items(), key=lambda row: int(row[0]))),
    }
```

In `build_safe_auto_candidates.py`:

- import `fusion_stats` and the three named default thresholds;
- change default output to `runs/safe-auto-candidates-v2`;
- change `schema_version` to `safe-auto-candidates-v2`;
- record the containment, normalized-center-distance, and maximum-area-ratio values under `config.dedup`;
- build `stats` by merging `fusion_stats(raw_candidates, fused)` with image and temporal counts;
- retain the existing refusal to overwrite an output directory and the before/after truth-hash check.

- [ ] **Step 4: Run all Python tests and verify GREEN**

Run:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
```

Expected: zero failures.

- [ ] **Step 5: Commit the auditable v2 artifact slice**

```powershell
git add -- ml/tests/test_auto_labeling.py ml/src/crrc_vision/auto_labeling.py ml/scripts/build_safe_auto_candidates.py
git commit -m "feat: report candidate dedup audit metrics"
```

### Task 4: External regeneration and acceptance gate

**Files:**
- Create outside Git: `E:/Work/京新数智/识动hicool/中车眼镜数据资产/runs/safe-auto-candidates-v2/candidates.json`
- Create outside Git: a v2 review-pack directory selected by the existing review-pack builder
- Modify: `PROJECT_STATUS.md`

- [ ] **Step 1: Hash formal truth immediately before regeneration**

Run:

```powershell
Get-FileHash 'E:\Work\京新数智\识动hicool\中车眼镜数据资产\annotations\fastener-v2\instances.json' -Algorithm SHA256
```

Expected hash: `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`.

- [ ] **Step 2: Recover the exact v1 input paths and regenerate to v2**

Read the v1 `input_hashes` keys and `config`, pass the same teacher/HSV inputs, manifest, source, truth, IoU, and temporal setting to `build_safe_auto_candidates.py`, and set only:

```text
--output runs/safe-auto-candidates-v2
```

Expected: command exits 0, creates a new directory, and prints v2 statistics. If the target exists, choose a new versioned output name rather than deleting or overwriting it.

- [ ] **Step 3: Compare v1/v2 on the two known failure scenes**

Use a read-only JSON query to report, for image names containing `0007` or `0047`, raw and fused candidate counts and cluster member counts before and after. Expected: v2 fused counts are lower on at least one known failure scene; neither scene gains candidates.

- [ ] **Step 4: Perform full-resolution entity-merge review**

Render v2 boxes on the original 0007 and 0047 images plus at least one adjacent-double-fastener scene. Inspect the entire image, not crops alone. Reject the gate if a fused box combines two independently identifiable physical fasteners; ambiguous cases remain separate or `uncertain`.

- [ ] **Step 5: Verify truth hash and full suite again**

Run:

```powershell
Get-FileHash 'E:\Work\京新数智\识动hicool\中车眼镜数据资产\annotations\fastener-v2\instances.json' -Algorithm SHA256
Set-Location ml
& .\.venv\Scripts\python.exe -m pytest -q
git diff --check
git status --short
```

Expected: truth hash remains the required value, pytest has zero failures, `git diff --check` is clean, and Git contains no private image or model artifact.

- [ ] **Step 6: Update durable project status and commit**

Record the v1/v2 counts, visual acceptance decision, immutable asset paths, truth hash, remaining 59-train/13-val deficit, and whether the 30-minute heartbeat can safely resume in `PROJECT_STATUS.md`.

```powershell
git add -- PROJECT_STATUS.md
git commit -m "docs: record candidate dedup validation"
```
