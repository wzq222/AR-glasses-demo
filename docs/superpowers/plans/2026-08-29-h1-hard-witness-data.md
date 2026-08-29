# H1真实困难防松标记数据 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在Git外生成并严格审核24张H1a真实风格防松标记难例，形成可追溯的局部状态数据、全图检测数据和下一阶段模型训练输入。

**Architecture:** 保留既有三态合成流水线不变，新增独立H1合同、固定配比job planner、ImageGen任务包、预标注、双层盲审和严格审计。每个ImageGen结果绑定真实train参考、prompt、拓扑、状态意图和SHA-256；局部通过后再嵌入真实train背景并做全图审核。

**Tech Stack:** Python 3.11、pytest、OpenCV、Pillow、COCO JSON、built-in ImageGen；所有图像和标注写入`E:/crrc_vision_data`。

---

## 文件边界

- `ml/src/crrc_vision/witness_state_contract.py`：H1状态、拓扑、mark role、记录合同与验证。
- `ml/src/crrc_vision/hard_sample_plan.py`：固定H1a配比、确定性job分配和prompt内容。
- `ml/src/crrc_vision/hard_sample_preannotation.py`：颜色mask、端点候选和需复核原因，不自动批准。
- `ml/src/crrc_vision/hard_sample_review.py`：局部/全图审核包、隐藏首审结论的二审合同和结论合并。
- `ml/src/crrc_vision/hard_sample_composite.py`：通过局部审核的样本嵌入真实train背景并变换标注。
- `ml/src/crrc_vision/hard_sample_audit.py`：配比、来源、哈希、几何、审核覆盖和formal truth硬门。
- `ml/scripts/build_h1_hard_sample_jobs.py`：建立参考包与24个ImageGen jobs。
- `ml/scripts/preannotate_h1_hard_samples.py`：为已生成图片建立候选sidecar。
- `ml/scripts/build_h1_review_pack.py`：建立局部或全图审核包。
- `ml/scripts/apply_h1_review.py`：应用哈希绑定审核结论。
- `ml/scripts/build_h1_full_images.py`：生成全图难例。
- `ml/scripts/audit_h1_hard_samples.py`：运行最终门并输出机器报告。
- `ml/tests/test_witness_state_contract.py`、`test_hard_sample_plan.py`、`test_hard_sample_preannotation.py`、`test_hard_sample_review.py`、`test_hard_sample_composite.py`、`test_hard_sample_audit.py`：单元和拒绝路径。

### Task 1: 建立H1状态与来源合同

**Files:**
- Create: `ml/src/crrc_vision/witness_state_contract.py`
- Create: `ml/tests/test_witness_state_contract.py`

- [ ] **Step 1: 写失败测试，固定状态、拓扑、负样本和train-only约束**

```python
from crrc_vision.witness_state_contract import validate_h1_record


def test_h1_record_requires_physical_state_contract() -> None:
    record = {
        "sample_id": "h1a-0001",
        "intent": "SUBTLE_DISPLACED",
        "output_state": "DISPLACED",
        "topology": "nut_plate",
        "mark_role": "bridges_moving_fixed",
        "has_marked_point": True,
        "source_split": "train",
        "eligible_split": "train",
        "source_scene_id": "scene-001",
        "source_reference_sha256": "A" * 64,
        "prompt_sha256": "B" * 64,
    }
    assert validate_h1_record(record) == ()


def test_lookalike_must_not_be_a_marked_point() -> None:
    record = {
        "sample_id": "h1a-0002",
        "intent": "LOOKALIKE",
        "output_state": None,
        "topology": "unknown",
        "mark_role": "ambiguous",
        "has_marked_point": True,
        "source_split": "train",
        "eligible_split": "train",
        "source_scene_id": "scene-002",
        "source_reference_sha256": "A" * 64,
        "prompt_sha256": "B" * 64,
    }
    assert "LOOKALIKE_MARKED_POINT_CONFLICT" in validate_h1_record(record)
```

- [ ] **Step 2: 运行测试并确认接口不存在**

Run: `.\.venv\Scripts\python.exe -m pytest ml/tests/test_witness_state_contract.py -q`

Expected: FAIL with `ModuleNotFoundError: crrc_vision.witness_state_contract`。

- [ ] **Step 3: 实现最小合同**

```python
from __future__ import annotations

from collections.abc import Mapping


H1_INTENTS = frozenset({
    "ALIGNED", "SUBTLE_DISPLACED", "OBVIOUS_DISPLACED",
    "DAMAGED_MARK", "INSUFFICIENT", "LOOKALIKE",
})
OUTPUT_STATES = frozenset({"ALIGNED", "DISPLACED", "DAMAGED_MARK", "INSUFFICIENT"})
TOPOLOGIES = frozenset({
    "bolt_head_plate", "nut_stud", "nut_plate", "double_nut",
    "fitting_pipe", "clamp_pipe", "unknown",
})
MARK_ROLES = frozenset({
    "bridges_moving_fixed", "moving_only", "fixed_only", "ambiguous",
})
INTENT_TO_STATE = {
    "ALIGNED": "ALIGNED",
    "SUBTLE_DISPLACED": "DISPLACED",
    "OBVIOUS_DISPLACED": "DISPLACED",
    "DAMAGED_MARK": "DAMAGED_MARK",
    "INSUFFICIENT": "INSUFFICIENT",
    "LOOKALIKE": None,
}


def _digest(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdefABCDEF" for char in text)


def validate_h1_record(record: Mapping[str, object]) -> tuple[str, ...]:
    errors: list[str] = []
    intent = str(record.get("intent", ""))
    if intent not in H1_INTENTS:
        errors.append("INVALID_INTENT")
    if record.get("output_state") != INTENT_TO_STATE.get(intent):
        errors.append("INTENT_STATE_MISMATCH")
    if str(record.get("topology", "")) not in TOPOLOGIES:
        errors.append("INVALID_TOPOLOGY")
    if str(record.get("mark_role", "")) not in MARK_ROLES:
        errors.append("INVALID_MARK_ROLE")
    if intent == "LOOKALIKE" and record.get("has_marked_point") is not False:
        errors.append("LOOKALIKE_MARKED_POINT_CONFLICT")
    if intent != "LOOKALIKE" and record.get("has_marked_point") is not True:
        errors.append("POSITIVE_MARKED_POINT_REQUIRED")
    if record.get("source_split") != "train" or record.get("eligible_split") != "train":
        errors.append("SYNTHETIC_MUST_BE_TRAIN_ONLY")
    for key in ("source_reference_sha256", "prompt_sha256"):
        if not _digest(record.get(key)):
            errors.append(f"INVALID_{key.upper()}")
    if not str(record.get("sample_id", "")).strip() or not str(record.get("source_scene_id", "")).strip():
        errors.append("MISSING_IDENTITY")
    return tuple(errors)
```

- [ ] **Step 4: 运行聚焦测试**

Run: `.\.venv\Scripts\python.exe -m pytest ml/tests/test_witness_state_contract.py -q`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add ml/src/crrc_vision/witness_state_contract.py ml/tests/test_witness_state_contract.py
git commit -m "feat: add hard witness state contract"
```

### Task 2: 建立确定性H1a任务计划与prompt包

**Files:**
- Create: `ml/src/crrc_vision/hard_sample_plan.py`
- Create: `ml/scripts/build_h1_hard_sample_jobs.py`
- Create: `ml/tests/test_hard_sample_plan.py`

- [ ] **Step 1: 写失败测试，固定24张配比和同源不重复状态**

```python
from collections import Counter

from crrc_vision.hard_sample_plan import build_h1a_jobs


def test_h1a_jobs_have_frozen_quota_and_lineage() -> None:
    references = [
        {
            "reference_id": f"ref-{index:02d}",
            "source_scene_id": f"scene-{index:02d}",
            "source_split": "train",
            "source_reference_sha256": f"{index:064X}",
            "crop_path": f"crops/ref-{index:02d}.png",
        }
        for index in range(1, 13)
    ]
    topology = {row["reference_id"]: "nut_plate" for row in references}
    jobs = build_h1a_jobs(references, topology, seed=20260829)
    assert len(jobs) == 24
    assert Counter(job["intent"] for job in jobs) == {
        "ALIGNED": 4,
        "SUBTLE_DISPLACED": 6,
        "OBVIOUS_DISPLACED": 4,
        "DAMAGED_MARK": 4,
        "INSUFFICIENT": 3,
        "LOOKALIKE": 3,
    }
    assert all(job["source_split"] == job["eligible_split"] == "train" for job in jobs)
    by_reference: dict[str, set[str]] = {}
    for job in jobs:
        by_reference.setdefault(job["reference_id"], set()).add(job["intent"])
    assert all(len(values) == 2 for values in by_reference.values())
```

- [ ] **Step 2: 运行测试确认RED**

Run: `.\.venv\Scripts\python.exe -m pytest ml/tests/test_hard_sample_plan.py -q`

Expected: FAIL with missing module/function。

- [ ] **Step 3: 实现固定配比、确定性分配和六类prompt**

实现`H1A_QUOTAS`、`build_h1a_jobs(references, topology_by_reference, seed)`和
`build_hard_sample_prompt(job)`。prompt必须包含以下共同不变量：

```python
COMMON = (
    "Create one photorealistic close-up phone inspection photo using the supplied real rail-equipment crop "
    "as the structural reference. Preserve the connection topology, camera viewpoint, metal geometry, grime, "
    "rust, oil, shadows and surrounding industrial context. No illustration, CGI, text, watermark, duplicate "
    "fasteners, melted geometry or floating components. Change only the state requested below."
)
```

状态片段固定为：

```python
STATE_PROMPTS = {
    "ALIGNED": "Keep the moving component physically fixed; the two paint fragments bridge both components and remain aligned, but add one realistic difficulty: glare, mild blur, partial shadow or worn paint.",
    "SUBTLE_DISPLACED": "Rotate only the moving component and its attached paint fragment by 2 to 8 degrees around the true joint axis; keep the fixed component and fixed-side paint unchanged; the result must be subtle but physically consistent.",
    "OBVIOUS_DISPLACED": "Rotate only the moving component and its attached paint fragment by 18 to 35 degrees around the true joint axis; keep the fixed component unchanged and preserve mechanically valid contact.",
    "DAMAGED_MARK": "Do not move either mechanical component; create irregular cured-paint chipping, cracking, fading or contamination that breaks visual continuity without a rigid relative rotation.",
    "INSUFFICIENT": "Keep the physical state ambiguous because one paint fragment is occluded, out of focus, overexposed or only one side is marked; do not invent a visible displacement.",
    "LOOKALIKE": "Remove any valid paint bridge and show a realistic red or yellow lookalike such as heat-shrink tubing, terminal sleeve, warning paint, rust or reflection; it must not be a marked inspection point.",
}
```

每个job写入`sample_id`、intent、output_state、topology、mark_role、reference路径/哈希、prompt及prompt
SHA-256。脚本必须读取外部`topology-decisions.json`，拒绝缺失、重复或非train来源，并在写入前后核验formal
truth。

- [ ] **Step 4: 运行聚焦和合同测试**

Run: `.\.venv\Scripts\python.exe -m pytest ml/tests/test_hard_sample_plan.py ml/tests/test_witness_state_contract.py -q`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add ml/src/crrc_vision/hard_sample_plan.py ml/scripts/build_h1_hard_sample_jobs.py ml/tests/test_hard_sample_plan.py
git commit -m "feat: build deterministic H1 image jobs"
```

### Task 3: 建立参考拓扑决策并生成H1a 24张局部图

**Files:**
- Git外 create: `E:/crrc_vision_data/synthetic/marked-point-h1/reference-pool/`
- Git外 create: `E:/crrc_vision_data/synthetic/marked-point-h1/topology-decisions.json`
- Git外 create: `E:/crrc_vision_data/synthetic/marked-point-h1/h1a/jobs.json`
- Git外 create: `E:/crrc_vision_data/synthetic/marked-point-h1/h1a/generated/`
- Git外 create: `E:/crrc_vision_data/synthetic/marked-point-h1/h1a/generation-manifest.json`

- [ ] **Step 1: 建立24个真实train参考池，再确定12个拓扑合格参考**

Run:

```powershell
.\.venv\Scripts\python.exe ml\scripts\build_synthetic_reference_pack.py `
  --data-root E:\crrc_vision_data `
  --reviewed-coco E:\crrc_vision_data\annotations\marked-point-v1.4\instances.train.json `
  --source-dir E:\crrc_vision_data\source\20240529-luosi `
  --output E:\crrc_vision_data\synthetic\marked-point-h1\reference-pool `
  --count 24
```

使用`view_image`检查24个reference crop和原始上下文，选择12个独立scene且拓扑可判断的参考。
`topology-decisions.json`格式固定：

```json
{
  "schema_version": "h1-topology-review-v1",
  "reviewer": "codex",
  "records": [
    {
      "reference_id": "ref-01",
      "topology": "nut_plate",
      "mark_role": "bridges_moving_fixed",
      "decision": "APPROVED",
      "reason": "paint crosses nut and fixed plate"
    }
  ]
}
```

模糊、单边涂点或无法辨认moving/fixed的参考可以保留给`INSUFFICIENT/LOOKALIKE`，但不得分配
`ALIGNED/DISPLACED`；无法确认任何样本角色的参考标为`REJECTED`。12个入选参考不足时重新以更低排名
但仍通过亮度/清晰度门的train场景补建reference pool，不得强制指定拓扑。

- [ ] **Step 2: 构建24个jobs并核对配比**

Run:

```powershell
$env:CRRC_VISION_DATA_ROOT='E:\crrc_vision_data'
.\.venv\Scripts\python.exe ml\scripts\build_h1_hard_sample_jobs.py `
  --references E:\crrc_vision_data\synthetic\marked-point-h1\reference-pool\references.json `
  --topology-decisions E:\crrc_vision_data\synthetic\marked-point-h1\topology-decisions.json `
  --output E:\crrc_vision_data\synthetic\marked-point-h1\h1a `
  --seed 20260829
```

Expected JSON: `jobs=24`以及固定`4/6/4/4/3/3`配比，formal truth哈希不变。

- [ ] **Step 3: 按job逐个调用built-in ImageGen**

每个job单独调用一次，参考图角色固定为`reference image`而非编辑目标；prompt使用jobs.json中的完整文本。
生成结果复制到`generated/<sample_id>-attempt-01.png`。若工具失败，仅重试相同job，不更改prompt和状态。

- [ ] **Step 4: 建立生成manifest**

每条记录必须保存：`sample_id`、`attempt`、`tool=built-in-imagegen`、reference/prompt/image SHA-256、生成文件
相对路径和`review_status=UNREVIEWED`。同一job最多3次尝试；尚未批准时不得把它计入24张完成集。

- [ ] **Step 5: 核验正式真值与文件数**

Run:

```powershell
(Get-FileHash E:\crrc_vision_data\annotations\fastener-v2\instances.json -Algorithm SHA256).Hash
(Get-ChildItem E:\crrc_vision_data\synthetic\marked-point-h1\h1a\generated -File).Count
```

Expected: truth为`B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`；首次尝试24张。

### Task 4: 建立不自动批准的H1预标注

**Files:**
- Create: `ml/src/crrc_vision/hard_sample_preannotation.py`
- Create: `ml/scripts/preannotate_h1_hard_samples.py`
- Create: `ml/tests/test_hard_sample_preannotation.py`

- [ ] **Step 1: 写失败测试，要求lookalike无正框且所有预标注为UNREVIEWED**

```python
import numpy as np

from crrc_vision.hard_sample_preannotation import preannotate_h1


def test_preannotation_never_self_approves() -> None:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[60:66, 30:98, 2] = 255
    result = preannotate_h1(image, intent="ALIGNED")
    assert result["review_status"] == "UNREVIEWED"
    assert result["paint_mask_pixels"] > 0


def test_lookalike_has_no_marked_point_box() -> None:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    result = preannotate_h1(image, intent="LOOKALIKE")
    assert result["has_marked_point"] is False
    assert result["bbox_xyxy"] is None
```

- [ ] **Step 2: 运行测试确认RED**

Run: `.\.venv\Scripts\python.exe -m pytest ml/tests/test_hard_sample_preannotation.py -q`

- [ ] **Step 3: 实现候选mask、bbox和几何提议**

复用`extract_witness_mark_mask`与`extract_witness_mark_geometry`，但捕获提取失败并写入
`review_reasons`。输出包含paint mask路径、候选bbox、fixed/moving端点、mask像素数、清晰度、亮度、
饱和像素比例；`DAMAGED_MARK/INSUFFICIENT`允许无有效两段线，`LOOKALIKE`强制无正框。任何状态均固定
`review_status=UNREVIEWED`。

- [ ] **Step 4: 运行聚焦测试并对24张生成sidecar**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest ml/tests/test_hard_sample_preannotation.py -q
.\.venv\Scripts\python.exe ml\scripts\preannotate_h1_hard_samples.py `
  --jobs E:\crrc_vision_data\synthetic\marked-point-h1\h1a\jobs.json `
  --generated E:\crrc_vision_data\synthetic\marked-point-h1\h1a\generated `
  --output E:\crrc_vision_data\synthetic\marked-point-h1\h1a\preannotations
```

Expected: 24条UNREVIEWED记录；不存在自动APPROVED。

- [ ] **Step 5: 提交**

```powershell
git add ml/src/crrc_vision/hard_sample_preannotation.py ml/scripts/preannotate_h1_hard_samples.py ml/tests/test_hard_sample_preannotation.py
git commit -m "feat: preannotate hard witness samples"
```

### Task 5: 局部首审与隐藏结论二审

**Files:**
- Create: `ml/src/crrc_vision/hard_sample_review.py`
- Create: `ml/scripts/build_h1_review_pack.py`
- Create: `ml/scripts/apply_h1_review.py`
- Create: `ml/tests/test_hard_sample_review.py`

- [ ] **Step 1: 写失败测试，要求全覆盖、哈希绑定及疑难类二审**

```python
from crrc_vision.hard_sample_review import validate_h1_reviews


def test_subtle_and_damaged_require_blind_second_review() -> None:
    manifest = {"records": [
        {"sample_id": "h1a-0001", "intent": "SUBTLE_DISPLACED", "image_sha256": "A" * 64}
    ]}
    first = {"records": [
        {"sample_id": "h1a-0001", "decision": "APPROVED", "image_sha256": "A" * 64}
    ]}
    assert "SECOND_REVIEW_MISSING:h1a-0001" in validate_h1_reviews(manifest, first, None)
```

- [ ] **Step 2: 运行测试确认RED**

Run: `.\.venv\Scripts\python.exe -m pytest ml/tests/test_hard_sample_review.py -q`

- [ ] **Step 3: 实现审核合同和审核包**

审核包每条同时提供原尺寸局部图、2倍nearest-neighbor细节图、paint-mask overlay和未带状态结论的metadata。
首审核对：机械拓扑、moving/fixed归属、状态意图、合成痕迹和线段端点。`SUBTLE_DISPLACED`、
`DAMAGED_MARK`及首审`UNCERTAIN`必须进入二审；二审包不得包含首审decision或reason。只有两轮一致或二审
明确解决时可APPROVED，冲突保留UNCERTAIN。

- [ ] **Step 4: Codex逐张完成首审和必要二审**

不能批准物理关系不成立、融化几何、整体旋转伪造状态、固定侧跟随移动、明显生成纹理或状态不可唯一解释
的图。被拒job用相同reference和intent生成下一attempt，直到固定配比各job有一个APPROVED结果或达到3次
尝试。

- [ ] **Step 5: 运行审核验证并提交代码**

Run: `.\.venv\Scripts\python.exe -m pytest ml/tests/test_hard_sample_review.py -q`

```powershell
git add ml/src/crrc_vision/hard_sample_review.py ml/scripts/build_h1_review_pack.py ml/scripts/apply_h1_review.py ml/tests/test_hard_sample_review.py
git commit -m "feat: add blind review for hard witness samples"
```

### Task 6: 嵌入真实全图并做受控退化

**Files:**
- Create: `ml/src/crrc_vision/hard_sample_composite.py`
- Create: `ml/scripts/build_h1_full_images.py`
- Create: `ml/tests/test_hard_sample_composite.py`

- [ ] **Step 1: 写失败测试，固定group、标注变换和lookalike负样本行为**

```python
from crrc_vision.hard_sample_composite import transform_h1_annotation


def test_full_transform_keeps_state_and_train_group() -> None:
    annotation = {
        "intent": "SUBTLE_DISPLACED",
        "output_state": "DISPLACED",
        "bbox_xyxy": [10, 20, 50, 60],
        "fixed_segment_xyxy": [[15, 40], [30, 40]],
        "moving_segment_xyxy": [[30, 40], [45, 43]],
        "source_scene_id": "scene-01",
    }
    result = transform_h1_annotation(annotation, scale=0.5, offset_xy=(100, 200))
    assert result["bbox_xyxy"] == [105, 210, 125, 230]
    assert result["output_state"] == "DISPLACED"
    assert result["scene_group"] == "scene-01"
```

- [ ] **Step 2: 运行测试确认RED**

Run: `.\.venv\Scripts\python.exe -m pytest ml/tests/test_hard_sample_composite.py -q`

- [ ] **Step 3: 实现全图合成**

局部样本只嵌入真实train背景中的已审核检查点区域；保持source scene group，不使用val/sealed-test。变换参数
由seed确定：目标短边4--40像素、曝光`0.75--1.25`、白平衡每通道`0.90--1.10`、运动模糊核`0/3/5`、
高斯失焦sigma`0--1.2`、JPEG quality`45--95`。所有参数写manifest。`LOOKALIKE`不写marked-point正框，
但保留hard-negative region；其余写局部状态、端点、topology和mask路径。

- [ ] **Step 4: 生成24张全图并核对group与尺寸分布**

Run:

```powershell
.\.venv\Scripts\python.exe ml\scripts\build_h1_full_images.py `
  --approved-local E:\crrc_vision_data\synthetic\marked-point-h1\h1a\local-approved.json `
  --background-coco E:\crrc_vision_data\annotations\marked-point-v1.4\instances.train.json `
  --source E:\crrc_vision_data\source\20240529-luosi `
  --output E:\crrc_vision_data\synthetic\marked-point-h1\h1a-full `
  --seed 20260829
```

Expected: 24张、0个非train来源、0个跨group泄漏、3个lookalike无正框。

- [ ] **Step 5: 提交**

```powershell
git add ml/src/crrc_vision/hard_sample_composite.py ml/scripts/build_h1_full_images.py ml/tests/test_hard_sample_composite.py
git commit -m "feat: composite hard witness full images"
```

### Task 7: 全图审核、最终审计和冻结H1a

**Files:**
- Create: `ml/src/crrc_vision/hard_sample_audit.py`
- Create: `ml/scripts/audit_h1_hard_samples.py`
- Create: `ml/tests/test_hard_sample_audit.py`
- Create: `docs/validation/2026-08-29-h1-hard-witness-data.md`
- Modify: `PROJECT_STATUS.md`

- [ ] **Step 1: 写失败测试，覆盖24张固定配比、审核、哈希和formal truth**

```python
from crrc_vision.hard_sample_audit import audit_h1_dataset


def test_h1_gate_rejects_missing_intent_quota(tmp_path) -> None:
    result = audit_h1_dataset(
        records=[{"sample_id": "only-one", "intent": "ALIGNED", "review_status": "APPROVED"}],
        asset_root=tmp_path,
        formal_truth_sha256="B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001",
    )
    assert result.passed is False
    assert "H1A_TOTAL:1:required=24" in result.errors
```

- [ ] **Step 2: 实现严格门**

最终门必须验证：固定配比`4/6/4/4/3/3`、24个独立job各一个APPROVED结果、局部与全图均审核、疑难类
二审覆盖、reference/prompt/local/full/mask/COCO哈希一致、端点/mask/bbox在界内、lookalike无正框、所有来源
train、同源group未泄漏、formal truth哈希固定、生成内容不在Git目录。

- [ ] **Step 3: 对24张全图逐张审核并应用结论**

全图审核必须在原始分辨率和缩放到现场目标尺寸两种视图下完成。检测目标不可见、融合边缘明显、尺寸不符合
真实分布或局部状态被压缩破坏时，拒绝该full placement并重新选择位置/参数；不能修改已批准局部状态来迁就
全图。

- [ ] **Step 4: 运行最终审计与全套测试**

Run:

```powershell
.\.venv\Scripts\python.exe ml\scripts\audit_h1_hard_samples.py `
  --manifest E:\crrc_vision_data\synthetic\marked-point-h1\h1a-full\manifest.json `
  --coco E:\crrc_vision_data\synthetic\marked-point-h1\h1a-full\instances.train.json `
  --review-pack E:\crrc_vision_data\synthetic\marked-point-h1\h1a-full-review\manifest.json `
  --formal-truth E:\crrc_vision_data\annotations\fastener-v2\instances.json
.\.venv\Scripts\python.exe -m pytest ml/tests -q
git diff --check
```

Expected: H1 gate PASS、24 approved、固定配比满足、253项既有测试加新增测试全部通过、formal truth哈希不变。

- [ ] **Step 5: 写验证证据、更新项目状态并提交**

验证文档记录全部输入/输出SHA-256、ImageGen尝试/拒绝/替换数、局部/全图/二审覆盖、各intent数量、拓扑
分布、变换分布、正式真值哈希和能力边界。明确H1a只证明数据管线通过，不证明真实松动准确率。

```powershell
git add ml/src/crrc_vision/hard_sample_audit.py ml/scripts/audit_h1_hard_samples.py `
  ml/tests/test_hard_sample_audit.py docs/validation/2026-08-29-h1-hard-witness-data.md PROJECT_STATUS.md
git commit -m "feat: freeze audited H1 hard witness data"
```

### Task 8: H1a通过后的扩展与状态模型切换门

**Files:**
- Git外 create: `E:/crrc_vision_data/synthetic/marked-point-h1/h1b/`
- Create after H1a PASS: `docs/superpowers/plans/2026-08-29-witness-state-model.md`

- [ ] **Step 1: 只有H1a最终门PASS才扩H1b**

使用同一合同和审核链补齐每类24张，总H1为120张；新增reference优先覆盖`double_nut`、`fitting_pipe`、
`clamp_pipe`、强反光、低照度、遮挡和密集相邻检查点。不得通过复制或普通像素增强凑数量。

- [ ] **Step 2: 统计H1真实拓扑和失败模式后再写模型计划**

状态模型计划必须基于实际通过审核的mask、端点、topology与图像尺寸，固定Fast-SCNN输入、loss、batch、三seed、
真实val门和ONNX导出；若没有真实受控`ALIGNED/DISPLACED`验证对，计划只能训练研究原型，禁止定义生产阈值。
