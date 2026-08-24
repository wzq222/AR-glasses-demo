# 中车眼镜三步巡检 MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在固定 CY01 眼镜与 Android 手机上完成二维码、防松线、万用表三步巡检，并输出可追溯的 JSON 与证据照片。

**Architecture:** 保留现有 `GlassBleService` 作为眼镜媒体通道，在独立 `inspection` 包中实现固定 SOP、分析器适配器和证据仓库。眼镜同步、手机相机和相册最终都只产出本地图片 URI；会话层不依赖具体媒体来源。二维码本地识别，防松线与万用表第一版走统一 HTTP 分析契约，低置信度进入人工确认。

**Tech Stack:** Android Java、JUnit 4、AndroidX Test、ZXing Core、OkHttp、现有 BLE/WiFi Direct/K900 SDK。

---

## 文件结构

```text
app/src/main/java/com/ar/glass/inspection/
├── model/          # 步骤、状态、结果和值对象
├── session/        # 固定三步状态机
├── analyzer/       # QR、本地/远程分析器与装配
├── evidence/       # JSON 和证据索引持久化
├── media/          # 眼镜、相机、相册的统一入口
└── ui/             # 巡检界面及 ViewModel/控制器
app/src/test/java/com/ar/glass/inspection/        # 纯 Java 单元测试
app/src/androidTest/java/com/ar/glass/inspection/ # Android 集成测试
docs/hardware/                                    # 设备与固件矩阵
docs/validation/                                  # 构建、实机、样本验收记录
```

## Task 1: 冻结可复现的硬件与构建基线

**Files:**
- Create: `docs/hardware/device-matrix.md`
- Create: `docs/validation/baseline-run.md`
- Modify: `PROJECT_STATUS.md`

- [ ] **Step 1: 记录基线提交与工具版本**

Run:

```powershell
git rev-parse HEAD
java -version
.\gradlew.bat --version
```

Expected: Git 提交为当前工作提交；Java 与 Gradle 版本完整写入 `baseline-run.md`。

- [ ] **Step 2: 记录设备矩阵**

`device-matrix.md` 每行必须包含：眼镜商品名、BLE 名称、硬件批次、固件、K900 SDK/AAR 哈希、手机型号、Android 版本、是否支持拍照/录像/补光/扬声器/麦克风。未知能力写“未实测”，并附负责人和验证日期，不能写成支持。

- [ ] **Step 3: 运行干净构建**

Run:

```powershell
.\gradlew.bat clean assembleDebug
Get-FileHash .\app\build\outputs\apk\debug\app-debug.apk -Algorithm SHA256
```

Expected: `BUILD SUCCESSFUL`，APK 哈希写入 `baseline-run.md`。

- [ ] **Step 4: 完成手机与眼镜复验**

在固定组合上执行 10 次“发现—连接—同步—选择照片—退出导入模式”，记录每次成功/失败、耗时、
照片数量和失败日志。成功、取消、断连三条路径结束后都必须能再次拍照。

- [ ] **Step 5: 提交基线证据**

```powershell
git add docs/hardware/device-matrix.md docs/validation/baseline-run.md PROJECT_STATUS.md
git commit -m "docs: freeze glasses hardware and build baseline"
```

## Task 2: 建立三步巡检域模型与状态机

**Files:**
- Create: `app/src/main/java/com/ar/glass/inspection/model/InspectionStep.java`
- Create: `app/src/main/java/com/ar/glass/inspection/model/AnalysisResult.java`
- Create: `app/src/main/java/com/ar/glass/inspection/session/SessionState.java`
- Create: `app/src/main/java/com/ar/glass/inspection/session/InspectionSession.java`
- Test: `app/src/test/java/com/ar/glass/inspection/session/InspectionSessionTest.java`
- Modify: `app/build.gradle`

- [ ] **Step 1: 加入 JUnit 依赖**

```gradle
testImplementation 'junit:junit:4.13.2'
```

- [ ] **Step 2: 写失败的状态机测试**

测试必须覆盖：初始步骤为 `QR`；成功后依次进入 `NUT_LINE`、`METER`；低置信度进入
`REVIEWING`；识别错误进入 `RETRYABLE_ERROR` 且不前进；人工确认后才保存；第三步保存后完成。

Run:

```powershell
.\gradlew.bat testDebugUnitTest --tests "com.ar.glass.inspection.session.InspectionSessionTest"
```

Expected: 类尚不存在导致编译失败。

- [ ] **Step 3: 实现固定领域契约**

`InspectionStep` 只允许：

```java
public enum InspectionStep { QR, NUT_LINE, METER }
```

`AnalysisResult` 使用不可变字段：`step`、`status`、`value`、`unit`、`confidence`、
`requiresHumanReview`、`capturedAt`、`imageUri`、`analyzerVersion`、`errorCode`。置信度范围固定为
`0.0..1.0`；失败结果必须有 `errorCode`；成功结果必须有 `imageUri` 和 `analyzerVersion`。

`InspectionSession` 对外只提供 `beginCapture()`、`submitImage(uri)`、`submitResult(result)`、
`confirmResult()`、`retry()`、`cancel()`，所有非法转换抛出 `IllegalStateException`。

- [ ] **Step 4: 运行状态机测试**

```powershell
.\gradlew.bat testDebugUnitTest --tests "com.ar.glass.inspection.session.InspectionSessionTest"
```

Expected: 全部通过。

- [ ] **Step 5: 提交领域层**

```powershell
git add app/build.gradle app/src/main/java/com/ar/glass/inspection app/src/test/java/com/ar/glass/inspection
git commit -m "feat: add fixed three-step inspection session"
```

## Task 3: 统一眼镜、手机相机和相册图片入口

**Files:**
- Create: `app/src/main/java/com/ar/glass/inspection/media/MediaSource.java`
- Create: `app/src/main/java/com/ar/glass/inspection/media/GalleryMediaSource.java`
- Create: `app/src/main/java/com/ar/glass/inspection/media/CameraMediaSource.java`
- Create: `app/src/main/java/com/ar/glass/inspection/media/GlassesMediaSource.java`
- Modify: `app/src/main/java/com/ar/glass/core/GlassBleService.java`
- Modify: `app/src/main/java/com/ar/glass/util/EventMsg.java`
- Test: `app/src/test/java/com/ar/glass/inspection/media/GlassesMediaSourceTest.java`

- [ ] **Step 1: 定义单一媒体契约**

```java
public interface MediaSource {
    void requestImage(Callback callback);
    interface Callback {
        void onImage(String localUri);
        void onError(String stableErrorCode, String message);
    }
}
```

三种来源都必须返回本地 URI，不把 Bitmap 长期保存在内存中。

- [ ] **Step 2: 写眼镜同步桥测试**

使用假的同步网关验证：只选择当前新照片；空列表返回 `GLASSES_NO_NEW_MEDIA`；取消返回
`GLASSES_IMPORT_CANCELLED`；无论成功或失败都调用一次退出导入模式。

- [ ] **Step 3: 从 BLE 服务暴露可测试网关**

把 `syncPhotos()`、`finalizeImport()`、`cancelSync()` 包装成最小 `GlassesSyncGateway`，现有协议与
下载实现保持不变。`GlassesMediaSource` 只订阅稳定事件，不解析 BLE 帧。

- [ ] **Step 4: 接入 CameraX/系统相机与相册 URI**

手机相机和相册走 Android Activity Result API；复制到 App 私有证据目录后再返回 URI，避免外部
授权失效。

- [ ] **Step 5: 运行测试与提交**

```powershell
.\gradlew.bat testDebugUnitTest
git add app/src/main/java/com/ar/glass app/src/test/java/com/ar/glass
git commit -m "feat: unify glasses camera and gallery media sources"
```

## Task 4: 实现二维码分析器与统一远程分析协议

**Files:**
- Create: `app/src/main/java/com/ar/glass/inspection/analyzer/StepAnalyzer.java`
- Create: `app/src/main/java/com/ar/glass/inspection/analyzer/QrAnalyzer.java`
- Create: `app/src/main/java/com/ar/glass/inspection/analyzer/RemoteAnalyzer.java`
- Create: `app/src/main/java/com/ar/glass/inspection/analyzer/AnalyzerRegistry.java`
- Test: `app/src/test/java/com/ar/glass/inspection/analyzer/QrAnalyzerTest.java`
- Test: `app/src/test/java/com/ar/glass/inspection/analyzer/RemoteAnalyzerTest.java`
- Modify: `app/build.gradle`

- [ ] **Step 1: 加入分析依赖**

```gradle
implementation 'com.google.zxing:core:3.5.3'
implementation 'com.squareup.okhttp3:okhttp:4.12.0'
testImplementation 'com.squareup.okhttp3:mockwebserver:4.12.0'
```

- [ ] **Step 2: 定义分析接口**

```java
public interface StepAnalyzer {
    InspectionStep supportedStep();
    AnalysisResult analyze(String localImageUri) throws AnalysisException;
    String version();
}
```

- [ ] **Step 3: 用生成二维码写失败测试并实现本地识别**

测试生成包含 `CRRC-DEMO-001` 的二维码，验证 `QrAnalyzer` 返回相同字符串、非空版本和原图 URI；
无二维码图片返回 `QR_NOT_FOUND`，不能返回空成功结果。

- [ ] **Step 4: 冻结防松线/万用表 HTTP 契约**

Request: `POST /v1/analyze/{nut-line|meter}`，multipart 字段仅含 `image`、`sessionId`、`stepId`。

Response:

```json
{
  "status": "SUCCEEDED",
  "value": "MISALIGNED",
  "unit": null,
  "confidence": 0.82,
  "modelVersion": "nut-line-yolov8-v1",
  "errorCode": null
}
```

`RemoteAnalyzerTest` 使用 MockWebServer 覆盖成功、HTTP 500、超时、非法 JSON、低置信度五条路径；
低置信度必须设置 `requiresHumanReview=true`。

- [ ] **Step 5: 执行测试与提交**

```powershell
.\gradlew.bat testDebugUnitTest --tests "com.ar.glass.inspection.analyzer.*"
git add app/build.gradle app/src/main/java/com/ar/glass/inspection/analyzer app/src/test/java/com/ar/glass/inspection/analyzer
git commit -m "feat: add QR and remote inspection analyzers"
```

## Task 5: 保存可追溯的巡检证据

**Files:**
- Create: `app/src/main/java/com/ar/glass/inspection/evidence/EvidenceRepository.java`
- Create: `app/src/main/java/com/ar/glass/inspection/evidence/JsonEvidenceRepository.java`
- Test: `app/src/test/java/com/ar/glass/inspection/evidence/JsonEvidenceRepositoryTest.java`

- [ ] **Step 1: 写序列化失败测试**

验证完整会话写入 `files/inspections/<session-id>/result.json`；JSON 含 session ID、开始/结束时间、
人员代号、点位码、硬件/固件、三个步骤结果、算法版本与证据 URI。失败步骤保留错误码；重复保存同一
session 使用临时文件加原子替换，不产生半份 JSON。

- [ ] **Step 2: 实现 App 私有目录证据仓库**

仓库复制三张证据图到 session 目录，原图复制成功后才允许清理同步临时目录。导出使用 Android
FileProvider，不申请 `MANAGE_EXTERNAL_STORAGE` 作为正常路径。

- [ ] **Step 3: 执行测试与提交**

```powershell
.\gradlew.bat testDebugUnitTest --tests "com.ar.glass.inspection.evidence.*"
git add app/src/main/java/com/ar/glass/inspection/evidence app/src/test/java/com/ar/glass/inspection/evidence
git commit -m "feat: persist inspection evidence as atomic JSON"
```

## Task 6: 接入巡检 UI 与失败恢复

**Files:**
- Create: `app/src/main/java/com/ar/glass/inspection/ui/InspectionActivity.java`
- Create: `app/src/main/res/layout/activity_inspection.xml`
- Modify: `app/src/main/java/com/ar/glass/ui/MainActivity.java`
- Modify: `app/src/main/AndroidManifest.xml`
- Test: `app/src/androidTest/java/com/ar/glass/inspection/ui/InspectionActivityTest.java`

- [ ] **Step 1: 写端到端仪器测试**

用假媒体源和假分析器验证：主界面进入巡检；当前步骤文案正确；每步选择图片后显示结果；低置信度
必须确认；错误可重拍；完成页能分享 JSON；取消流程会关闭眼镜导入模式。

- [ ] **Step 2: 实现单屏三步 UI**

界面只显示当前任务、步骤进度、采集来源、证据缩略图、结果、重拍/确认按钮和稳定错误提示。
所有用户可见文本写入 `strings.xml`，不得显示调试日志或算法内部推理。

- [ ] **Step 3: 清理权限路径**

验证 Android 12/13 的 BLE、附近 WiFi、通知、相机权限；正常证据存储使用 App 私有目录和
FileProvider。若 `MANAGE_EXTERNAL_STORAGE` 不再需要，从 Manifest 删除。

- [ ] **Step 4: 执行回归并提交**

```powershell
.\gradlew.bat testDebugUnitTest connectedDebugAndroidTest assembleDebug
git add app/src/main app/src/androidTest
git commit -m "feat: deliver three-step inspection workflow"
```

## Task 7: 完成算法与实机验收门

**Files:**
- Create: `docs/validation/sample-set-manifest.csv`
- Create: `docs/validation/mvp-acceptance.md`
- Modify: `PROJECT_STATUS.md`

- [ ] **Step 1: 建立受控样本清单**

每类至少覆盖正常、异常、暗光、反光、模糊和无目标；CSV 记录样本哈希、来源授权、真值、拍摄距离、
光照、设备/固件，不把敏感原图提交到公共仓库。

- [ ] **Step 2: 验证真实防松线与万用表服务**

算法负责人提供符合 Task 4 契约的版本化地址；在受控集上保存逐样本结果、混淆矩阵、失败样本和延迟。
演示门槛由业务方在 `mvp-acceptance.md` 明确签字，未签字时只报告测得数值，不自行发明准确率承诺。

- [ ] **Step 3: 连续三轮实机 SOP**

三轮都要从连接开始，经三步采集、识别、人工确认/重拍、JSON 导出，到眼镜恢复拍照结束。记录每轮
视频、结果 JSON、APK 哈希和服务版本。

- [ ] **Step 4: 更新项目状态并提交**

```powershell
git add docs/validation PROJECT_STATUS.md
git commit -m "test: record MVP field acceptance evidence"
```

## 计划自检

- 规格覆盖：固定三步、统一图片入口、三项分析、人工复核、证据留痕、失败恢复、实机验收均有任务。
- 外部边界：模型训练、服务器部署、后台用户系统、内窥镜和跨品牌迁移不混入本客户端计划。
- 完成定义：构建通过只证明工程可构建；只有 Task 7 的实机与样本证据通过后，才可称巡检 MVP 完成。
