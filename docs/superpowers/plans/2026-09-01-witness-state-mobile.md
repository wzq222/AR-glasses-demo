# Witness State Mobile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不伪造生产松动结论的前提下，交付手机端可运行的防松线 ROI 状态测量链：输出角度区间、临时复核提示和拒判原因，并为真实标定及轻量模型训练建立可审计接口。

**Architecture:** 继续用全图检测器寻找候选；候选 ROI 交给 MobileNetV3-Small 多头模型估计固定侧、活动侧、防松线与接缝证据；确定性几何求解器计算角度及 95% 区间；双阈值只生成 `review_hint`，未标定时正式 `FastenerState` 始终为 `INSUFFICIENT`。合成样本只验证几何重建，真实状态能力必须由受控成对采集标定。

**Tech Stack:** Python 3.11、pytest、PyTorch/torchvision（训练脚本运行时依赖）、ONNX；Android Java、JUnit、ONNX Runtime、CameraX。

---

## Task 1: Python 临时分诊与置信区间合同

**Files:**

- Modify: `ml/src/crrc_vision/witness_state.py`
- Modify: `ml/tests/test_witness_state.py`

- [x] 先增加失败测试，覆盖 `3°`、`15°` 精确边界、跨阈值区间、第二帧确认、未标定正式状态保持 `INSUFFICIENT`、无效区间拒判。
- [x] 运行 `./.venv/Scripts/python.exe -m pytest ml/tests/test_witness_state.py -q`，确认新测试因缺少区间分诊接口而失败。
- [x] 新增 `AngleInterval`、`ProvisionalTriageThresholds` 和独立 `triage_witness_angle`；输出只使用 `LIKELY_ALIGNED / POSSIBLE_DISPLACED / LIKELY_DISPLACED` 提示，不扩展正式四态。
- [x] 调整 `evaluate_witness_state`：在阈值未标定时仍计算可用几何并附带临时提示，但正式状态固定 `INSUFFICIENT`。
- [x] 重跑目标测试和全量 Python 测试。
- [x] Commit: `feat(ml): add fail-closed witness angle triage`

## Task 2: 健康历史基准控制限

**Files:**

- Create: `ml/src/crrc_vision/witness_baseline.py`
- Create: `ml/tests/test_witness_baseline.py`

- [x] 先写失败测试，覆盖 `max(3°, median(abs(delta)) + 3 * 1.4826 * MAD)`、乱序输入、非有限数值、样本不足和极端离群点。
- [x] 运行目标测试并确认缺少模块而失败。
- [x] 实现不可变 `BaselineControlLimit` 与估计器；至少 5 次健康重复拍摄才产生阈值，保留样本数、中位数、MAD 和原因。
- [x] 重跑目标测试和全量 Python 测试。
- [x] Commit: `feat(ml): add robust witness baseline limit`

## Task 3: ROI 状态数据合同与 Git 外清单

**Files:**

- Create: `ml/src/crrc_vision/witness_roi_dataset.py`
- Create: `ml/tests/test_witness_roi_dataset.py`
- Create: `ml/scripts/build_witness_roi_dataset.py`

- [x] 先写失败测试，覆盖 24 张合成样本的相对路径、图像/掩膜/端点哈希、`eligible_split=train`、场景隔离、重复 ID 和 formal truth 哈希门。
- [x] 运行目标测试并确认缺少实现而失败。
- [x] 实现只读导入 `repositioned-approved-v2/approved-locals.json` 的构建器；输出必须在 Git 外，原始 formal truth 只验哈希不修改。
- [x] 写出可复现 manifest，显式标记 `synthetic_geometry_only=true`、`real_state_truth=false` 和 `sealed_test_opened=false`。
- [x] 用 `E:/crrc_vision_data` 实跑一次，记录样本数、状态分布和 formal truth 哈希。
- [x] Commit: `feat(ml): build audited witness roi manifest`

## Task 4: 轻量 ROI 多头基线与合成几何验证

**Files:**

- Create: `ml/src/crrc_vision/witness_roi_model.py`
- Create: `ml/tests/test_witness_roi_model.py`
- Create: `ml/scripts/train_witness_roi_model.py`
- Create: `ml/scripts/export_witness_roi_onnx.py`

- [x] 先写失败测试，固定输入 `N×3×320×320`，验证四类掩膜 logits、四个关键点热图、质量 logits 的形状和非有限数拒绝。
- [x] 实现 MobileNetV3-Small 共享骨干和轻量解码头；训练库只在脚本执行时导入，避免破坏基础包。
- [x] 使用轻度透视、缩放、模糊、亮度与颜色抖动；几何标签同步变换，禁止会改变松动角真值的独立随机旋转。
- [x] 在 24 张合成 train-only 样本上做过拟合冒烟，报告关键点/角度重建误差，不报告真实准确率。
- [x] 导出动态 batch 的 ONNX，验证输出名、形状、模型哈希及 Python ONNX 冒烟；质量门未过，`android_packaging_allowed=false`。
- [x] Commit: `feat(ml): train lightweight witness roi baseline`

## Task 5: Android 分诊合同与 Python 对齐

**Files:**

- Create: `app/src/main/java/com/ar/glass/vision/fastener/WitnessReviewHint.java`
- Create: `app/src/main/java/com/ar/glass/vision/fastener/AngleInterval.java`
- Create: `app/src/main/java/com/ar/glass/vision/fastener/ProvisionalTriageThresholds.java`
- Modify: `app/src/main/java/com/ar/glass/vision/fastener/GeometryDecision.java`
- Modify: `app/src/main/java/com/ar/glass/vision/fastener/AntiLooseGeometry.java`
- Modify: `app/src/test/java/com/ar/glass/vision/fastener/AntiLooseGeometryTest.java`

- [x] 先写 Android 失败测试，复刻 Python 的 3°/15° 边界、跨界区间、第二帧和未标定拒判用例。
- [x] 运行 `./gradlew.bat testDebugUnitTest --tests com.ar.glass.vision.fastener.AntiLooseGeometryTest`，确认缺少合同而失败。
- [x] 实现 Java 值对象和分诊逻辑；`GeometryDecision.state` 在未标定时保持 `INSUFFICIENT`，新增 hint、点估计与上下界。
- [x] 增加 Python/Java 共享 JSON 向量，逐条核对路由一致。
- [x] 重跑 Android 目标测试和全量单测。
- [x] Commit: `feat(android): mirror fail-closed witness triage`

## Task 6: Android ROI ONNX 推理与候选交互

> 2026-09-01阻塞：实验模型质量门FAIL，`android_packaging_allowed=false`。按安全设计不把失败模型
> 接入候选点击或安装到手机；待独立物理点角度门通过后继续本任务。

**Files:**

- Create: `app/src/main/java/com/ar/glass/vision/realtime/OnnxWitnessStateEstimator.java`
- Create: `app/src/main/java/com/ar/glass/vision/realtime/WitnessRoiTransform.java`
- Create: `app/src/test/java/com/ar/glass/vision/realtime/WitnessRoiTransformTest.java`
- Modify: `app/src/main/java/com/ar/glass/ui/DetectionOverlayView.java`
- Modify: `app/src/main/java/com/ar/glass/ui/LiveInspectionActivity.java`
- Modify: `app/src/main/res/layout/activity_live_inspection.xml`
- Modify: `app/src/main/res/values/strings.xml`
- Modify: `app/build.gradle`

- [ ] 先写 ROI 扩框/裁剪/坐标回投测试，覆盖边界框、竖图、横图和空框。
- [ ] 加载 Git 外导出的状态 ONNX；模型缺失、输出错形、非有限值和异常全部回落 `INSUFFICIENT`。
- [ ] 候选点击后才运行单 ROI；展示角度区间、灰/绿/黄/红复核提示和耗时，所有文案保留“待确认”。
- [ ] 增加“确认正常 / 确认错位 / 无法确认重拍”本地记录合同；未经人工确认不得写正式状态。
- [ ] 构建 Debug APK，在 P20 Pro 验证单 ROI 冷/热 P50、P95、内存和 10 分钟稳定性。
- [ ] Commit: `feat(android): run witness roi review on selected candidate`

## Task 7: 验证收口与下一训练门

**Files:**

- Create: `docs/validation/2026-09-01-witness-state-mobile-baseline.md`
- Modify: `README.md`
- Modify: `PROJECT_STATUS.md`

- [x] 运行全量 Python、Android 单测和 APK 构建。
- [x] 对 formal truth 重算 SHA-256，必须仍为 `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`。
- [x] 记录模型/APK 哈希、合成角度误差、Android 时延未测原因、拒判边界和仍缺的真实受控状态组。
- [x] 明确区分“链路可用”“合成几何可重建”“真实松动准确率已验证”；没有独立真实测试时不得宣称后者。
- [x] Commit: `docs: validate witness state mobile baseline`

## Separate Follow-up: 全图检测器轻量化

当前 43MB YOLOv8s-P2 的蒸馏不与状态 ROI 训练混在同一次实验中。状态链完成后，单独建立
YOLOv8n-P2 学生模型计划，以当前 S-P2 为教师，使用原图+切片、定向 hard-negative mining 和独立
跨设备场景门；只有候选召回达到门槛才替换手机模型。
