# 中车眼镜全图防松线 Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从482张现场全图建立可审计的预标注数据、训练首个全图候选检测模型，并在Android工程中交付可测试的多目标结果与防松线几何判定核心。

**Architecture:** 现场资产保存在Git外的 `E:/Work/京新数智/识动hicool/中车眼镜数据资产`；仓库中的 `ml/` 只保存可复现的数据审计、预标注、复核和训练入口。Android本轮先实现模型无关的结构化契约、切片/坐标/几何组件；真实ncnn运行时在验证模型导出后接入。

**Tech Stack:** Python 3.12、Pillow、OpenCV、NumPy、pytest、PyTorch CUDA、COCO JSON、D-FINE-N、Android Java 8、JUnit 4。

---

## 文件结构

```text
ml/
├── pyproject.toml
├── README.md
├── src/crrc_vision/
│   ├── assets.py          # 私有资产根目录与清单
│   ├── inventory.py       # 图片尺寸、哈希、时间与质量统计
│   ├── grouping.py        # 连拍/近重复场景分组与split
│   ├── prelabel.py        # 防松色标候选与框/关键点预标注
│   ├── coco.py            # COCO输出与校验
│   └── review.py          # 叠加图和复核索引
├── scripts/
│   ├── bootstrap_assets.py
│   ├── build_prelabels.py
│   ├── build_review_pack.py
│   └── train_dfine.py
└── tests/
    ├── test_assets.py
    ├── test_inventory.py
    ├── test_grouping.py
    ├── test_prelabel.py
    └── test_coco.py

app/src/main/java/com/ar/glass/vision/fastener/
├── VisionPoint.java
├── FastenerState.java
├── FastenerInspection.java
├── FullImageInspectionResult.java
├── GeometryThresholds.java
├── GeometryDecision.java
└── AntiLooseGeometry.java

app/src/test/java/com/ar/glass/vision/fastener/
└── AntiLooseGeometryTest.java
```

## Task 1: 建立私有资产根与Python包

**Files:**
- Modify: `.gitignore`
- Create: `ml/pyproject.toml`
- Create: `ml/README.md`
- Create: `ml/src/crrc_vision/__init__.py`
- Create: `ml/src/crrc_vision/assets.py`
- Create: `ml/tests/test_assets.py`
- Create outside Git: `E:/Work/京新数智/识动hicool/中车眼镜数据资产/`

- [ ] **Step 1: 配置隔离环境与依赖清单**

`.gitignore` 增加 `.venv/`、`ml/.pytest_cache/`、`ml/**/*.egg-info/`、`ml/.cache/` 和模型扩展名；
`pyproject.toml` 固定Pillow、NumPy、OpenCV headless、imagehash、pytest与包入口。然后运行：

```powershell
& 'C:\Users\holdo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e "ml[dev]" -i https://mirrors.aliyun.com/pypi/simple/
```

- [ ] **Step 2: 写资产根失败测试**

```python
def test_asset_root_requires_environment(monkeypatch):
    monkeypatch.delenv("CRRC_VISION_DATA_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="CRRC_VISION_DATA_ROOT"):
        asset_root()

def test_asset_root_resolves_existing_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("CRRC_VISION_DATA_ROOT", str(tmp_path))
    assert asset_root() == tmp_path.resolve()
```

- [ ] **Step 3: 验证RED**

Run: `.\.venv\Scripts\python.exe -m pytest ml/tests/test_assets.py -v`

Expected: FAIL，因为 `crrc_vision.assets` 尚不存在。

- [ ] **Step 4: 实现最小资产契约**

```python
def asset_root() -> Path:
    value = os.environ.get("CRRC_VISION_DATA_ROOT")
    if not value:
        raise RuntimeError("CRRC_VISION_DATA_ROOT is not set")
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"CRRC_VISION_DATA_ROOT does not exist: {root}")
    return root
```

- [ ] **Step 5: 创建资产目录并复制原图**

Run:

```powershell
$root='E:\Work\京新数智\识动hicool\中车眼镜数据资产'
New-Item -ItemType Directory -Force -Path "$root\source\20240529-luosi","$root\annotations\prelabel-v1","$root\annotations\reviewed-v1","$root\splits","$root\review-packs","$root\runs","$root\models","$root\exports\android","$root\packages"
Copy-Item -LiteralPath 'C:\Users\holdo\AppData\Local\Temp\codex-luosi-20260825\luosi\*' -Destination "$root\source\20240529-luosi" -Force
```

Expected: 源目录与目标目录均为482张图片；不删除临时源。

- [ ] **Step 6: 验证GREEN并提交**

Run: `.\.venv\Scripts\python.exe -m pytest ml/tests/test_assets.py -v`

Expected: 2 passed。

Commit: `feat: bootstrap private vision asset workspace`

## Task 2: 数据审计与场景隔离拆分

**Files:**
- Create: `ml/src/crrc_vision/inventory.py`
- Create: `ml/src/crrc_vision/grouping.py`
- Create: `ml/tests/test_inventory.py`
- Create: `ml/tests/test_grouping.py`
- Create: `ml/scripts/bootstrap_assets.py`

- [ ] **Step 1: 写图片清单与分组失败测试**

```python
def test_scan_images_records_hash_dimensions_and_capture_time(tmp_path):
    image = tmp_path / "IMG_20240529_111456.jpg"
    Image.new("RGB", (20, 10), "red").save(image)
    row = scan_images(tmp_path)[0]
    assert row.width == 20 and row.height == 10
    assert len(row.sha256) == 64
    assert row.captured_at.isoformat() == "2024-05-29T11:14:56"

def test_near_duplicates_never_cross_splits():
    rows = fake_rows(times=[0, 1, 40], hashes=[0, 1, (1 << 63)])
    groups = group_scenes(rows, max_gap_seconds=3, max_hash_distance=4)
    splits = split_groups(groups, train_ratio=0.5)
    assert split_for(rows[0], splits) == split_for(rows[1], splits)
```

- [ ] **Step 2: 验证RED**

Run: `python -m pytest ml/tests/test_inventory.py ml/tests/test_grouping.py -v`

Expected: FAIL，因为扫描和分组函数尚不存在。

- [ ] **Step 3: 实现清单与确定性分组**

`scan_images`记录相对路径、SHA-256、尺寸、文件名时间、pHash和灰度拉普拉斯方差；
`group_scenes`按时间相邻不超过3秒或pHash汉明距离不超过4建立并查集；`split_groups`只在组级别按
固定seed分配80%训练、20%内部验证，不生成test。

- [ ] **Step 4: 生成真实资产清单**

Run:

```powershell
$env:CRRC_VISION_DATA_ROOT='E:\Work\京新数智\识动hicool\中车眼镜数据资产'
python ml/scripts/bootstrap_assets.py --source source/20240529-luosi
```

Expected: 生成 `manifest.jsonl`、`splits/phase1.json` 和 `runs/data-audit-v1.json`，统计482张、
2000×1500、0个精确重复，并输出场景组数量。

- [ ] **Step 5: 验证GREEN并提交**

Run: `python -m pytest ml/tests/test_inventory.py ml/tests/test_grouping.py -v`

Expected: 全部通过。

Commit: `feat: audit and group fastener image dataset`

## Task 3: 防松色标预标注与COCO输出

**Files:**
- Create: `ml/src/crrc_vision/prelabel.py`
- Create: `ml/src/crrc_vision/coco.py`
- Create: `ml/tests/test_prelabel.py`
- Create: `ml/tests/test_coco.py`
- Create: `ml/scripts/build_prelabels.py`

- [ ] **Step 1: 写合成图失败测试**

```python
def test_red_mark_produces_one_candidate_box():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.line(image, (130, 100), (170, 100), (0, 0, 255), 6)
    candidates = find_marked_fasteners(image, min_mark_area=20)
    assert len(candidates) == 1
    assert candidates[0].bbox.contains(150, 100)
    assert candidates[0].line.confidence > 0.5

def test_coco_rejects_box_outside_image():
    with pytest.raises(ValueError, match="outside image"):
        validate_annotation(width=100, height=100, bbox=[90, 90, 20, 20])
```

- [ ] **Step 2: 验证RED**

Run: `python -m pytest ml/tests/test_prelabel.py ml/tests/test_coco.py -v`

Expected: FAIL，因为候选与COCO模块尚不存在。

- [ ] **Step 3: 实现颜色/线段候选**

`find_marked_fasteners`把BGR转HSV，合并红色两段色相与黄色范围，执行3×3开运算和5×5闭运算；
过滤面积小于20或长宽均小于4的连通域；用 `cv2.fitLine` 得到两端点；以标记区域中心为基准按
短边的12%且不小于160像素生成正方形候选框并裁剪到图像边界。重叠候选IoU大于0.35时合并。

- [ ] **Step 4: 输出预标注**

Run:

```powershell
$env:CRRC_VISION_DATA_ROOT='E:\Work\京新数智\识动hicool\中车眼镜数据资产'
python ml/scripts/build_prelabels.py --manifest manifest.jsonl --output annotations/prelabel-v1/instances.json
```

Expected: COCO JSON中只有 `marked_fastener` 一类；每个annotation包含bbox、颜色标记端点、算法版本、
候选置信度和 `review_status=unreviewed`。

- [ ] **Step 5: 验证GREEN并提交**

Run: `python -m pytest ml/tests/test_prelabel.py ml/tests/test_coco.py -v`

Commit: `feat: generate auditable fastener prelabels`

## Task 4: 生成复核包并完成AI抽样复核

**Files:**
- Create: `ml/src/crrc_vision/review.py`
- Create: `ml/tests/test_review.py`
- Create: `ml/scripts/build_review_pack.py`
- Create: `docs/validation/2026-08-25-prelabel-audit.md`

- [ ] **Step 1: 写叠加图失败测试**

```python
def test_render_overlay_preserves_image_size():
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    rendered = render_overlay(image, [candidate(10, 20, 30, 40)])
    assert rendered.shape == image.shape
    assert np.any(rendered != image)
```

- [ ] **Step 2: 验证RED、实现并验证GREEN**

Run: `python -m pytest ml/tests/test_review.py -v`

Expected RED: `render_overlay`不存在。实现固定颜色框、端点、置信度和文件名水印后重跑，Expected GREEN:
1 passed。

- [ ] **Step 3: 生成分层复核包**

按场景组、候选数、置信度、清晰度分层选取至少60张；每张输出原图缩略图与叠加图，另生成
`review-index.csv`，字段为image、candidate_id、decision、corrected_bbox、comment。

- [ ] **Step 4: AI视觉复核**

逐张查看60张叠加图，填写 `accept`、`reject` 或 `needs_manual`。统计候选级精确率、图像级漏标观察值、
按暗光/反光/管接头分层的失败原因。若accept精确率低于0.80，不进入训练；先修订预标注器。

- [ ] **Step 5: 提交审计结果**

仓库只提交不含图片的 `docs/validation/2026-08-25-prelabel-audit.md`；复核图片和CSV留在数据资产目录。

Commit: `test: audit fastener prelabel quality`

## Task 5: Android多目标契约与几何判定

**Files:**
- Modify: `app/build.gradle`
- Create: `app/src/main/java/com/ar/glass/vision/fastener/VisionPoint.java`
- Create: `app/src/main/java/com/ar/glass/vision/fastener/FastenerState.java`
- Create: `app/src/main/java/com/ar/glass/vision/fastener/FastenerInspection.java`
- Create: `app/src/main/java/com/ar/glass/vision/fastener/FullImageInspectionResult.java`
- Create: `app/src/main/java/com/ar/glass/vision/fastener/GeometryThresholds.java`
- Create: `app/src/main/java/com/ar/glass/vision/fastener/GeometryDecision.java`
- Create: `app/src/main/java/com/ar/glass/vision/fastener/AntiLooseGeometry.java`
- Create: `app/src/test/java/com/ar/glass/vision/fastener/AntiLooseGeometryTest.java`

- [ ] **Step 1: 加入JUnit并写失败测试**

```java
@Test public void alignedSegmentsAreNormal() {
    GeometryDecision result = AntiLooseGeometry.evaluate(
        p(0, 0, 0.95f), p(10, 0, 0.95f), p(12, 0, 0.95f), p(22, 0, 0.95f), 100f,
        new GeometryThresholds(10f, 0.05f, 0.05f));
    assertEquals(FastenerState.NORMAL, result.getState());
}

@Test public void lowConfidenceIsUncertain() {
    GeometryDecision result = AntiLooseGeometry.evaluate(
        p(0, 0, 0.4f), p(10, 0, 0.95f), p(12, 0, 0.95f), p(22, 0, 0.95f), 100f,
        new GeometryThresholds(10f, 0.05f, 0.05f));
    assertEquals(FastenerState.UNCERTAIN, result.getState());
    assertEquals("KEYPOINT_CONFIDENCE_LOW", result.getReason());
}
```

- [ ] **Step 2: 验证RED**

Run: `.\gradlew.bat testDebugUnitTest --tests "com.ar.glass.vision.fastener.AntiLooseGeometryTest"`

Expected: 编译失败，因为类型尚不存在。

- [ ] **Step 3: 实现结构化契约与几何公式**

`AntiLooseGeometry.evaluate`计算无向夹角、归一化端点间距和两段中点到共同拟合直线残差；任一点置信度
低于0.60返回UNCERTAIN；在没有现场阈值文件时，只允许单元测试通过显式 `GeometryThresholds`
注入阈值，生产默认返回UNCERTAIN，避免把论文阈值伪装成现场真值。

- [ ] **Step 4: 验证GREEN并提交**

Run: `.\gradlew.bat testDebugUnitTest assembleDebug`

Expected: 新增测试通过，APK构建成功。

Commit: `feat: add full-image fastener result contract`

## Task 6: 首轮训练可行性门

**Files:**
- Create: `ml/scripts/train_dfine.py`
- Create: `ml/tests/test_training_gate.py`
- Create: `docs/validation/2026-08-25-training-readiness.md`

- [ ] **Step 1: 写训练门失败测试**

```python
def test_training_gate_rejects_unreviewed_annotations():
    report = TrainingReadiness(images=482, accepted=0, rejected=0, unreviewed=482)
    assert report.can_train is False
    assert "unreviewed" in report.reasons[0]

def test_training_gate_accepts_reviewed_dataset():
    report = TrainingReadiness(images=80, accepted=75, rejected=5, unreviewed=0)
    assert report.can_train is True
```

- [ ] **Step 2: 验证RED、实现训练门并验证GREEN**

Run: `python -m pytest ml/tests/test_training_gate.py -v`

- [ ] **Step 3: 建立项目虚拟环境**

使用bundled Python 3.12创建 `.venv`；PyPI依赖使用阿里云镜像；PyTorch/torchvision从PyTorch官方CUDA
wheel索引安装。安装后验证 `torch.cuda.is_available()` 和GPU名称为RTX 3060 Laptop GPU。

- [ ] **Step 4: 达门后启动D-FINE-N训练**

仅当Task 4复核精确率不低于0.80且所有训练annotation均为accepted时，从官方Apache-2.0 D-FINE固定
提交创建外部工具checkout，生成单类COCO配置；输入640、AMP、batch 4、梯度累积4、最多20 epoch，
每epoch保存验证AP和漏检清单。训练产物写入 `中车眼镜数据资产/runs/dfine-n-v1`。

- [ ] **Step 5: 记录结果**

若训练启动，记录checkpoint哈希、最佳epoch、内部验证AP、每场景漏检和ONNX导出检查；若训练门未过，
报告具体拒绝原因和需要补审的样本数，不生成误导性模型。

Commit: `feat: add guarded detector training entrypoint`

## Task 7: 全量验证与项目状态回写

**Files:**
- Modify: `README.md`
- Modify: `PROJECT_STATUS.md`
- Create: `docs/validation/2026-08-25-full-image-phase1.md`

- [ ] **Step 1: 运行Python和Android全量测试**

Run:

```powershell
$env:CRRC_VISION_DATA_ROOT='E:\Work\京新数智\识动hicool\中车眼镜数据资产'
python -m pytest ml/tests -v
.\gradlew.bat testDebugUnitTest assembleDebug
git diff --check
```

- [ ] **Step 2: 验证私有资产没有进入Git**

Run: `git status --short` 和 `git ls-files | Select-String -Pattern '\.(jpg|jpeg|png|pth|onnx|param|bin)$'`

Expected: 仅原有UI图标PNG被跟踪；无现场图片和模型文件。

- [ ] **Step 3: 写回真实状态**

README增加 `ml/` 使用入口和私有资产根说明；PROJECT_STATUS区分代码验证、标注质量、训练状态和真机状态；
验证报告包含命令、结果、数据清单哈希、限制和下一步。

- [ ] **Step 4: 提交**

Commit: `docs: record full-image vision phase one evidence`

## 计划自检

- 规格覆盖：私有资产、场景拆分、预标注、复核、训练门、Android多目标契约、几何不确定性和验证均有任务。
- 数据诚实性：没有受控松动真值时不训练或声称松动分类准确率。
- 类型一致性：Python只负责bbox/keypoints与训练；Android统一使用 `FullImageInspectionResult` 和
  `FastenerInspection`，旧布尔接口不承载多目标结果。
- 许可证：生产路径不依赖DINOv3、DEIMv2或Ultralytics；D-FINE只使用Apache-2.0官方代码并固定提交。
