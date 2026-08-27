# Teacher-Anchor Candidate Deduplication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the rejected complete-link v2 fusion with deterministic teacher-anchor grouping and unambiguous HSV marker assignment, then validate a new immutable v2.1 asset.

**Architecture:** Group candidates by image. Build `reference_teacher` anchor clusters with one-pass representative linkage, freeze their weighted boxes, then attach other candidates only to a unique matching anchor. HSV windows may match by their center point being within a 5%-expanded anchor; multiple eligible anchors require a normalized center-distance margin of 0.10, otherwise the candidate remains independent.

**Tech Stack:** Python 3.11+, pytest, existing CRRC candidate builder and Git-external asset store.

---

### Task 1: Teacher representative clusters

**Files:**
- Modify: `ml/tests/test_auto_labeling.py`
- Modify: `ml/src/crrc_vision/auto_labeling.py`

- [ ] **Step 1: Add the failing representative-variance test and strengthen the chain test**

```python
def test_teacher_variance_matches_cluster_representative() -> None:
    fused = fuse_candidates([
        candidate("a", "reference_teacher", "fastener", (0, 0, 20, 20), 0.8),
        candidate("b", "reference_teacher", "fastener", (4, 0, 24, 20), 0.8),
        candidate("c", "reference_teacher", "fastener", (7, 0, 27, 20), 0.8),
    ])
    assert len(fused) == 1


def test_representative_link_prevents_teacher_chain_bridge() -> None:
    fused = fuse_candidates([
        candidate("a", "reference_teacher", "fastener", (0, 0, 20, 20), 0.8),
        candidate("b", "reference_teacher", "fastener", (5, 0, 25, 20), 0.8),
        candidate("c", "reference_teacher", "fastener", (10, 0, 30, 20), 0.8),
    ])
    assert sorted(len(item.member_ids) for item in fused) == [1, 2]
```

- [ ] **Step 2: Verify RED**

Run the two tests. Expected: variance test fails with 2 clusters; chain test passes.

- [ ] **Step 3: Implement one-pass representative clustering**

Replace `_complete_link_cluster` with `_representative_cluster`. For every deterministically sorted pending row, compare it with `_weighted_box(cluster)` at that point. Do not revisit rejected rows, which prevents transitive rescan drift. Group rows by `relative_path` before clustering.

- [ ] **Step 4: Verify GREEN and run all auto-labeling tests**

Run `python -m pytest tests/test_auto_labeling.py -v`. Expected: zero failures.

### Task 2: Unique HSV marker-to-anchor assignment

**Files:**
- Modify: `ml/tests/test_auto_labeling.py`
- Modify: `ml/src/crrc_vision/auto_labeling.py`

- [ ] **Step 1: Add failing unique, ambiguous, and cross-category marker tests**

```python
def test_hsv_marker_center_uniquely_attaches_to_teacher_anchor() -> None:
    fused = fuse_candidates([
        candidate("teacher", "reference_teacher", "fastener", (0, 0, 20, 20), 0.9),
        candidate("marker", "hsv", "fastener", (-80, -80, 100, 100), 0.8),
    ])
    assert len(fused) == 1
    assert fused[0].supporting_families == ("hsv", "reference_teacher")


def test_hsv_marker_between_two_teacher_anchors_remains_independent() -> None:
    fused = fuse_candidates([
        candidate("left", "reference_teacher", "fastener", (0, 0, 20, 20), 0.9),
        candidate("right", "reference_teacher", "fastener", (22, 0, 42, 20), 0.9),
        candidate("marker", "hsv", "fastener", (-69, -80, 111, 100), 0.8),
    ])
    assert len(fused) == 3


def test_hsv_marker_cross_category_attachment_is_conflict() -> None:
    fused = fuse_candidates([
        candidate("teacher", "reference_teacher", "pipe_joint", (0, 0, 20, 20), 0.9),
        candidate("marker", "hsv", "fastener", (-80, -80, 100, 100), 0.8),
    ])
    assert len(fused) == 1
    assert fused[0].category is None
    assert fused[0].consensus_status == "conflict"
```

- [ ] **Step 2: Verify RED**

Run the three tests. Expected: unique and conflict tests fail because the large HSV window is not geometrically merged; ambiguous test already passes.

- [ ] **Step 3: Implement unique anchor selection**

Add constants:

```python
DEFAULT_HSV_ANCHOR_EXPANSION = 0.05
DEFAULT_ANCHOR_ASSIGNMENT_MARGIN = 0.10
```

For each image, cluster teacher rows first and freeze each teacher cluster's weighted box. An HSV row is eligible for an anchor when its window-center point-to-box distance divided by anchor diagonal is at most `0.05`; any row remains eligible through the existing composite geometric predicate. Rank eligible anchors by candidate-center to anchor-center distance divided by anchor diagonal. Attach if there is one eligible anchor, or if the second-best score minus best score is at least `0.10`; otherwise put it into the residual set. Geometrically cluster residual rows separately.

- [ ] **Step 4: Verify GREEN and determinism**

Run all auto-labeling tests twice, including the reversed-input test. Expected: identical IDs and zero failures.

- [ ] **Step 5: Commit Tasks 1-2**

```powershell
git add -- ml/tests/test_auto_labeling.py ml/src/crrc_vision/auto_labeling.py
git commit -m "fix: anchor HSV markers to physical candidates"
```

### Task 3: Immutable v2.1 audit artifact

**Files:**
- Modify: `ml/scripts/build_safe_auto_candidates.py`
- Modify: `ml/tests/test_auto_labeling.py`

- [ ] **Step 1: Extend the statistics/config expectation**

Assert the two new assignment constants retain `0.05` and `0.10`. This makes the safety thresholds explicit and reviewable.

- [ ] **Step 2: Change builder metadata only after the test exists**

Change default output and schema to `safe-auto-candidates-v2.1`. Record `strategy: teacher-anchor`, `hsv_anchor_expansion: 0.05`, `anchor_assignment_margin: 0.10`, and `linkage: representative-one-pass` in `config.dedup`. Keep v1/v2 assets untouched.

- [ ] **Step 3: Run the full Python suite and commit**

Run `python -m pytest -q` and `git diff --check`; expected zero failures/errors. Commit as `feat: audit teacher-anchor candidate fusion`.

### Task 4: Real-data acceptance

**Files:**
- Create outside Git: `runs/safe-auto-candidates-v2.1/candidates.json`
- Modify: `PROJECT_STATUS.md`

- [ ] **Step 1: Verify truth hash, recover v1 inputs, and generate v2.1**

Expected truth hash before and after: `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`. Use the v1 teacher/HSV inputs and generation configuration. Never delete an existing output.

- [ ] **Step 2: Gate on known scenes and an adjacent-object counterexample**

Require image 0007 and 0047 counts to decrease from 15 and 10. Render full-resolution overlays for both plus an image containing adjacent independently identifiable fasteners. Reject if any fused cluster spans two physical entities; uncertain assignments stay independent.

- [ ] **Step 3: Verify repository and update status**

Run the full Python suite, `git diff --check`, `git status --short`, and a tracked-file scan for private images/models. Record v1/v2/v2.1 counts, visual result, truth hash, remaining data deficit, and heartbeat decision in `PROJECT_STATUS.md`, then commit.

