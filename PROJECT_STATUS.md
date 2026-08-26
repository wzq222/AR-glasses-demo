# Project Status

## Purpose

“中车眼镜”是面向轨道交通设备巡检的 AR 眼镜辅助作业项目，近期目标是在固定硬件和受控场景下
跑通三步巡检 SOP：二维码打卡、防松线状态判断、万用表读数，并形成带时间与证据照片的结果记录。

## Verified Current State

- 2026-08-25：仓库从 `https://github.com/wzq222/AR-glasses-demo.git` 克隆，保留上游 Git 历史；
  基线提交为 `510e5f5`。
- 代码包含单个 Android 手机端模块，使用 Java、BLE、WiFi Direct、HTTP 和 K900 SDK。
- `GlassBleService` 实现眼镜发现/连接、进入照片导入模式、WiFi Direct 组网、HTTP 下载和退出导入模式。
- `ImageAnalyzer` 定义二维码、防松线、万用表三个接口，但 `DefaultImageAnalyzer` 均为占位实现；
  当前 UI 没有调用 `Vision`。
- 2026-08-25：482张现场全图已在Git外完成SHA-256/尺寸/清晰度清单，按时间与pHash划为177个场景组；
  训练424张、内部验证58张，场景泄漏0。
- `ml/` 已实现可复现的清单、分组、HSV色标候选、COCO输出、分层复核包和训练质量门。
- HSV候选v2生成1,993个候选；AI逐候选复核60张/257个候选，接受151、拒绝97、需人工9，精确率
  60.89%，低于80%训练门；D-FINE-N训练未启动。
- Android已加入全图多目标结构和防松线几何公式；关键点低置信度或现场阈值未标定时强制返回
  `UNCERTAIN`。这部分是模型无关核心，尚未接入真实检测推理。
- 2026-08-25：V2从177个场景组确定性选择100个代表帧（train 80/val 20），生成物理紧固件真值
  骨架；当前100张全部未复核、0个真值框，训练门继续拒绝。
- GroundingDINO 12张试跑产生33个建议框，逐框复核仅接受7、拒绝26，精确率21.21%，且12/12张
  存在明显漏框；试跑门为FAIL，未扩到100张，也未污染真值。
- 用户提供的Git外部参考权重确认为YOLOv8s三类检测模型；同域12图在640输入产生92框且候选质量较高，
  但两张不同机柜实拍仅产生2/0框，跨场景漏检严重。该权重只作私有标注教师；原数据、类别YAML、
  第二阶段分割权重和来源授权缺失，且存在Ultralytics AGPL-3.0/商业许可约束。
- 2026-08-26：Git外YOLOv8s参考教师已在100张V2代表帧上受限加载并完成推理，产生731个候选；
  AI逐页复核31页候选裁剪和13页整图叠加，候选层691接受、39需人工、1拒绝，但整图层100/100均
  需人工补全、6张零检测。正式真值SHA-256前后不变，训练门保持关闭。
- 2026-08-26：安全自动标注Phase A代码链路已实现：teacher多尺度整图/2×2重叠切片、HSV独立锚点、
  同场景ORB/RANSAC传播硬门、多来源融合、Codex首轮整图复核与盲审第二遍、整图银标质量门和拒绝式
  导出。72项Python测试通过；用旧100图teacher输出对482图清单做拒绝性实测时正确报告缺382图、
  不创建半成品目录，正式真值SHA-256保持不变。
- 2026-08-26：482图的640/960/1280整图+2×2切片teacher全量推理已完成，产生15,463个原始框；
  与1,993个HSV候选及1个通过硬门的时序候选合并后得到5,535个融合候选。ORB/RANSAC检查305条
  同场景相邻边，仅34条通过几何门，最终只传播1个有高共识来源的框。Git外Codex复核包已生成
  482张整图、5,535张候选上下文和61个首轮批次；22张图仍为零候选，必须做整图审核。
- 仓库没有眼镜端 App、后台服务、SOP 引擎、登录、巡检记录、语音引导、内窥镜接入、自动化测试或 CI。
- 2026-08-25：在 Windows 中文路径下加入 `android.overridePathCheck=true` 后，
  `.\gradlew.bat assembleDebug` 构建成功；APK 大小 28,268,821 bytes，SHA-256 为
  `D9B54B8C9A2402FA3FFC83C32229397A08112610EAE4C28D522D122851263E20`。
- 2026-08-25：已加入Python数据工具测试和Android几何JUnit测试；本机测试与APK构建证据见
  `docs/validation/2026-08-25-full-image-phase1.md`。
- Phase 1 debug APK为28,325,108 bytes，SHA-256
  `F795FA260905DAF85B387EF3EE1A106C4EAB7B2593A6CFA9410EB23CB46431F6`。
- 尚未完成真实手机/眼镜验证。

## Active Work

全图防松线Phase 1代码底座已实现；标注质量门正确拒绝训练。路线V2已把生产主线调整为
`PicoDet-S/M + 全图上下文 + 重叠切片`的物理紧固件检测，随后只在ROI内做色标关键点和几何判断。
当前工作重心是执行61批Codex首轮整图/候选审核，并对新增或调整框执行隐藏首轮结论的第二遍。
只有银标门达到至少64个train和16个val完整场景组，才进入
PicoDet-S/M训练与目标手机端P95基准；教师候选只用于提速，不承担整图完整性判断。
RTMDet-tiny-P2和D-FINE-N仅作为离线对照。

## Run

```powershell
.\gradlew.bat assembleDebug
$env:CRRC_VISION_DATA_ROOT='E:\Work\京新数智\识动hicool\中车眼镜数据资产'
.\.venv\Scripts\python.exe -m pytest ml/tests -v
```

APK 预期输出：`app/build/outputs/apk/debug/app-debug.apk`。

## Validate

```powershell
.\gradlew.bat clean assembleDebug
.\.venv\Scripts\python.exe -m pytest ml/tests -v
git status --short
```

实机验收另按 `docs/analysis/2026-08-25-demo-gap-analysis.md` 的 P0 验收门槛执行。

## Known Risks

- 当前代码来自一次性上游提交，且没有 LICENSE；代码、K900 AAR 和 native 库授权边界未确认。
- 会议没有冻结眼镜准确型号、固件、手机型号、协议版本和“rocket”品牌转写。
- 三种识别算法缺模型、样本、阈值和可量化验收指标。
- 当前482张来自同一约44分钟采集，缺跨车辆、跨手机、跨光照和独立测试集；没有受控“正常/松动”真值。
- 色标候选仍受锈蚀、警示贴、强光和断裂涂线影响，不能作为训练真值；当前训练门为FAIL。
- 通用开放词汇模型会把波纹管、滤筒和整束管线误判为连接件，且严重漏掉小紧固件；不能作为本批
  数据的批量自动标注器。
- 参考YOLOv8s教师虽然候选质量高，但100张整图复核均未证明完整，且6张完全零检测；类别名只能从
  视觉推断、原始YAML和授权缺失，因此不能直接回写正式真值或充当生产模型。
- 眼镜照片被导入后可能不再出现在 `media.config`，失败重试和证据保全存在数据丢失风险。
- 明文 HTTP、广泛存储权限和现场影像合规尚未形成交付方案。

## Next Smallest Action

执行Git外`review-packs/safe-auto-v1/first-pass`的61批Codex审核，对新增/调整框生成隐藏首轮结论的
第二轮任务并合并不可变决策清单。若银标门拒绝，则保留拒绝报告并补数据；若达到至少64个train和
16个val完整场景组，再并行训练PicoDet-S与M。
同时在指定手机与CY01眼镜上复验BLE连接和照片同步。

## Evidence

- `app/src/main/java/com/ar/glass/core/GlassBleService.java`：当前连接与照片同步实现。
- `app/src/main/java/com/ar/glass/vision/DefaultImageAnalyzer.java`：三个算法仍为占位。
- `docs/sources/2026-08-24-AR眼镜开发周会-逐字稿.txt`：会议需求来源。
- `docs/analysis/2026-08-25-demo-gap-analysis.md`：现状与目标差距及优先级。
- `docs/validation/2026-08-25-local-build.md`：本机路径修复、构建与测试证据。
- `docs/validation/2026-08-25-prelabel-audit.md`：60张/257候选的质量审计。
- `docs/validation/2026-08-25-training-readiness.md`：训练拒绝条件和达门后的固定训练方案。
- `docs/analysis/2026-08-25-full-image-fastener-route-v2.md`：移动端小目标路线重评与新验收门。
- `docs/validation/2026-08-25-full-image-v2-bootstrap.md`：代表帧、真值门与开放词汇试跑证据。
- Git外`review-packs/fastener-v2/reference-teacher-v1/ai-review-v1.json`：100图教师候选与整图复核结果。
