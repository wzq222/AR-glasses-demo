# 中车眼镜

面向轨道交通现场巡检的 AR 眼镜辅助作业项目。当前代码基线来自
[`wzq222/AR-glasses-demo`](https://github.com/wzq222/AR-glasses-demo)，现阶段已实现的主体是
CY01 眼镜与 Android 手机之间的 BLE 连接、WiFi Direct 组网和照片同步。防松线方向现已加入全图
数据审计、候选预标注、质量门、多目标结果契约和几何判定核心，银标训练门已通过并有一个
内部P2准确率基线，并已接入独立的手机相机测试页；ROI状态实验模型和3°/15°安全分诊合同已实现，
但实验质量门失败，尚未接入手机，也不能可靠判断真实松动状态。二维码、
万用表识别仍只有接口，三类能力都还没有完成真机作业闭环。

## 项目入口

- [项目状态](PROJECT_STATUS.md)
- [会议需求基线](docs/requirements/2026-08-24-meeting-requirements.md)
- [现有 Demo 缺口分析](docs/analysis/2026-08-25-demo-gap-analysis.md)
- [两周 MVP 设计](docs/superpowers/specs/2026-08-25-crrc-glasses-mvp-design.md)
- [实施计划](docs/superpowers/plans/2026-08-25-crrc-glasses-mvp.md)
- [全图防松线设计](docs/superpowers/specs/2026-08-25-full-image-fastener-vision-design.md)
- [预标注质量审计](docs/validation/2026-08-25-prelabel-audit.md)
- [训练就绪判断](docs/validation/2026-08-25-training-readiness.md)
- [全图防松线检测路线 V2](docs/analysis/2026-08-25-full-image-fastener-route-v2.md)
- [V2标注引导链路验证](docs/validation/2026-08-25-full-image-v2-bootstrap.md)
- [小目标准确率恢复验证](docs/validation/2026-08-28-small-object-accuracy-recovery.md)
- [高准确率严格验证收口](docs/validation/2026-08-28-high-accuracy-validation.md)
- [带防松标记检查点候选门](docs/validation/2026-08-29-marked-point-candidate-recall.md)
- [ImageGen 防松线合成试点](docs/validation/2026-08-29-synthetic-marked-point-pilot.md)
- [Android 手机相机候选测试验收](docs/validation/2026-09-01-android-phone-live-test.md)
- [防松线手机状态基线验证](docs/validation/2026-09-01-witness-state-mobile-baseline.md)
- [会议逐字稿](docs/sources/2026-08-24-AR眼镜开发周会-逐字稿.txt)

> 当前边界：不要把下方“图像识别接口（预留）”理解为算法已经接入。硬件实机、固件版本、
> 现场图像和准确率均未在本仓库中提供验证证据。

## 上游 Demo 说明

一款针对 **CY01 智能 AR 眼镜** 的 Android 应用。通过 BLE（低功耗蓝牙）连接眼镜，使用 **WiFi Direct + HTTP** 把眼镜拍摄的照片同步到手机，支持选择性导入、原图库浏览，并预留了图像识别扩展接口。

## 功能特性

- **多眼镜手动选择连接**：扫描并列出附近所有 CY01 眼镜，由用户手动选择要连接的设备，避免多设备同时开机时的连接冲突。
- **一键照片同步**：点击「同步照片到手机」后自动建立 WiFi Direct 连接，通过 HTTP 拉取眼镜照片并下载到手机。
- **选择性导入（带缩略图）**：下载完成后弹出勾选列表，显示每张照片的缩放图，用户可勾选要导入的照片；选中的移动到相册，未选中的自动清理。
- **原图库浏览**：网格 + 全屏两种方式浏览已同步的照片（保存在 `glass_media/photos` 目录）。
- **手机候选测试页**：手机后置相机已接入640输入YOLOv8s-P2 ONNX并叠加候选框；P20 Pro上单次
  冷机推理约1.9秒，采用推理后1秒冷却时约0.33次/秒更新，仅供内部测试，松动状态仍固定拒判。
- **严格验证结论**：S/M-P2 均未通过 `P>=0.90` 的召回与完整场景门；封存测试保持未打开。
- **标记点候选闭环**：Git外开发真值为30 train/17 val场景、248框，候选覆盖248/248；
  仍因train场景少于64且缺正常/松动状态真值而禁止训练或声称可判断松动。
- **状态实验基线**：24张合成train-only ROI可训练并导出约1.4M参数的MobileNetV3-Small多头ONNX，
  但合成角度mean/P95为3.37°/6.27°、mask IoU为0.136，质量门FAIL，Android打包关闭。
- **其他图像接口（预留）**：二维码识别和电压表数字识别待实现。

## 技术栈

- 原生 Android（Java），minSdk 21 / targetSdk 33
- BLE（Bluetooth Low Energy）连接与控制
- WiFi Direct（Wi-Fi P2P）数据传输
- HTTP 明文文件传输（原生 `HttpURLConnection`）
- EventBus（`org.greenrobot.eventbus`）实现服务与 UI 通信
- K900 SDK（`ksdk-release.aar`）：OTA、底层蓝牙封装等
- 内置 native 库：ONNX Runtime、Sherpa-ONNX、NiuTrans（推理/翻译相关）

## 项目结构

```
ar_glass_app/
├── app/
│   ├── src/main/
│   │   ├── java/com/ar/glass/
│   │   │   ├── core/                         # 核心逻辑
│   │   │   │   ├── GlassBleService.java      # BLE + WiFi Direct + HTTP 同步核心服务
│   │   │   │   ├── AppState.java             # 全局状态（眼镜 IP 等）
│   │   │   │   └── App.java                  # Application
│   │   │   ├── ui/                           # 界面
│   │   │   │   ├── MainActivity.java         # 主界面：连接、同步入口
│   │   │   │   ├── GalleryActivity.java      # 原图库（网格浏览）
│   │   │   │   └── ImageViewerActivity.java  # 全屏看图
│   │   │   ├── util/EventMsg.java            # EventBus 事件定义
│   │   │   └── vision/                       # 图像识别接口（预留）
│   │   │       ├── ImageAnalyzer.java        # 识别接口
│   │   │       ├── DefaultImageAnalyzer.java # 占位实现
│   │   │       └── Vision.java               # 识别引擎入口
│   │   ├── res/                              # 布局、图标、字符串等资源
│   │   ├── jniLibs/                          # native 库（onnxruntime、sherpa-onnx 等）
│   │   ├── libs/                             # K900 SDK（ksdk-release.aar 等）
│   │   └── AndroidManifest.xml
│   └── build.gradle
├── ml/                                      # 私有数据审计、预标注、复核与训练门
├── build.gradle
├── settings.gradle
└── gradle/                                   # Gradle Wrapper
```

## 构建与运行

1. 用 Android Studio 打开 `ar_glass_app` 目录。
2. 等待 Gradle 同步完成（自动下载依赖）。
3. 连接 Android 手机（Android 5.0 / API 21 及以上），点击 Run 编译安装。

命令行构建：

```bash
./gradlew assembleDebug
```

生成的 APK 位于 `app/build/outputs/apk/debug/`。

## 核心实现说明

### BLE 连接

CY01 使用两套 BLE 服务：

| 通道 | 服务 UUID | 说明 |
|------|-----------|------|
| NUS 控制通道 | `6e40fff0-b5a3-f393-e0a9-e50e24dcca9e` | 写特征 `6e400002-...`，通知特征 `6e400003-...` |
| 串口数据/文件通道 | `de5bf728-d711-4e47-af26-65e3012a5dc7` | 写特征 `de5bf72a-...`，通知特征 `de5bf729-...` |

串口大数据帧格式：

```
[0xBC][action][len低][len高][CRC16低][CRC16高][payload...]
```

### 照片同步流程

1. 用户点击「同步照片到手机」。
2. 发送 BLE 命令 `glassesControl(new byte[]{2,1,4,1})`，让眼镜进入照片导入模式。
3. 建立 WiFi Direct 连接，通过 BLE `action=115`（type=0x08）数据帧解析出眼镜 IP。
4. HTTP 拉取照片列表：`http://<ip>/files/media.config`（兜底 `http://<ip>:80/storage/sd0/C/DCIM/1/vf_list.txt`）。
5. 把全部照片下载到临时目录 `glass_media/tmp`，弹窗显示缩略图供勾选。
6. 用户勾选后，选中照片移动到 `glass_media/photos`，未选中的从临时目录删除。
7. 发送 `glassesControl(new byte[]{2,1,9})` 退出导入模式（否则眼镜无法继续拍照）。

### 图像识别接口（预留）

位于 `com.ar.glass.vision` 包，定义三个能力：

- `decodeQrCode(Bitmap)` —— 二维码识别，返回二维码内容
- `analyzeAntiLooseState(Bitmap)` —— 防松标记相对位移四态结果
- `isNutLoose(Bitmap)` —— 仅供旧调用兼容；会丢失损坏/拒判语义，新代码禁用
- `readMeterValue(Bitmap)` —— 电压表数字识别

默认实现 `DefaultImageAnalyzer` 为占位实现，待接入 ZXing / ML Kit / OCR 等算法；UI 层统一通过 `Vision.get()` 调用。

### 全图防松线开发入口

现场图片、复核图和模型不进入Git，默认放在
`E:/Work/京新数智/识动hicool/中车眼镜数据资产`，并通过环境变量显式指定：

```powershell
$env:CRRC_VISION_DATA_ROOT='E:\Work\京新数智\识动hicool\中车眼镜数据资产'
.\.venv\Scripts\python.exe -m pytest ml/tests -v
.\.venv\Scripts\python.exe ml\scripts\bootstrap_assets.py
.\.venv\Scripts\python.exe ml\scripts\build_prelabels.py
.\.venv\Scripts\python.exe ml\scripts\build_review_pack.py --output review-packs/prelabel-v2
.\.venv\Scripts\python.exe ml\scripts\build_fastener_selection.py
.\.venv\Scripts\python.exe ml\scripts\build_fastener_label_pack.py
.\.venv\Scripts\python.exe ml\scripts\train_dfine.py
```

最后一条命令是强制质量门：只要仍有未复核标注或抽检精确率低于80%，就拒绝训练。Android侧的
`com.ar.glass.vision.fastener` 已定义全图多目标结果和几何判定；没有现场标定阈值时只返回
`INSUFFICIENT`。正式状态为`ALIGNED / DISPLACED / DAMAGED_MARK / INSUFFICIENT`；当前不代表
检测或状态模型已经接入默认Android分析器。

## 依赖

- `androidx.appcompat` / `com.google.android.material` / `androidx.constraintlayout`
- `org.greenrobot:eventbus:3.3.1`
- `io.netty:netty-all`
- `com.google.protobuf:protobuf-javalite`
- `org.slf4j`
- `ksdk-release.aar`（K900 SDK）

## 注意事项

- 眼镜端 `media.config` 只列出「未传输」照片。照片一旦导入，眼镜会将其标记为已传输，之后无法再通过 `media.config` 列出（属眼镜固件行为，App 端无法改变）。
- 应用使用明文 HTTP 传输照片，已在 Manifest 中开启 `android:usesCleartextTraffic="true"`。
- 同步结束后必须发送退出导入模式命令 `glassesControl(new byte[]{2,1,9})`，否则眼镜会停留在占用状态、无法拍照。
