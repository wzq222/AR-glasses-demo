# AR 眼镜控制（CY01）

一款针对 **CY01 智能 AR 眼镜** 的 Android 应用。通过 BLE（低功耗蓝牙）连接眼镜，实现**照片同步**、**语音控制拍照**、**二维码识别**、**YOLO 目标检测**、**万用表读数识别**、**电量显示**、**语音播报**等能力。

## 功能特性

### 中车巡检任务
- 登录生产SOP后台并仅加载当前巡检员的任务。
- 按版本化流程逐步执行二维码、防松标记和万用表检查。
- 每步支持手机拍照、手机图库或 App 原图库选图；三种来源复用同一自动分析、人工结论、原始证据上传和幂等重试链路。
- 全部必做步骤、照片证据和人工确认齐全后才能提交后台复核。

### 设备连接
- **持续扫描**：应用启动后自动、不间断地扫描附近 CY01 眼镜，直到连接成功；手机蓝牙未开启时弹窗提示，开启后自动继续搜索。
- **多设备分区选择**：设备列表分为「已连接 / 已配对 / 已发现」三个分区——已连接为当前设备，已配对为系统历史配对记录（含关机设备），已发现为扫描到但尚未配对的新设备。
- **经典蓝牙配对**：BLE 连接成功后自动触发经典蓝牙配对，使眼镜的音频通道（A2DP / SCO）可用。

### 照片同步
- **一键同步**：点击「同步照片到手机」后自动建立 WiFi Direct 连接，通过 HTTP 拉取眼镜照片并下载。
- **选择性导入（带缩略图）**：下载完成后弹出勾选列表，显示每张照片缩略图；选中照片移动到相册，未选中自动清理。
- **原图库浏览**：网格 + 全屏两种方式浏览已同步照片（保存在 `glass_media/photos` 目录）。

### 语音控制与播报
- **语音拍照**：按住「按住说话」按钮，通过蓝牙 SCO 采集眼镜麦克风，松手后调用讯飞 SparkChain 在线语音听写（ASR）识别；识别文本经同音字容错表匹配后按意图分流——「二维码拍照」→ 二维码识别、「对齐拍照」→ YOLO 检测、「万用表拍照」→ 读数识别，普通「拍照」仅存原图库。
- **结果播报**：二维码识别结果通过 TTS 走媒体音频流（A2DP）从眼镜扬声器播报，未识别到也会语音提示。
- **录音提示音**：录音开始 / 结束各有「滴」提示音；连眼镜时走眼镜扬声器，未连接时走手机扬声器。

### 二维码识别
- 基于 ML Kit Barcode Scanning，采用**四级递进策略**（原图 / 对比度增强 / 放大 2 倍 / 放大 2 倍 + 对比度增强），提升远距离与模糊二维码的识别率。

### 防松标记检测与辅助判断
- 默认使用 512 输入的单类 YOLOv8s-P2 NCNN FP32 模型做全图候选，再由 128 输入的
  MobileNetV3-Small verifier 剔除非防松标记检查点。
- 对保留检查点运行 `witness-roi.onnx`，定位防松线掩膜、固定侧/活动侧两段线和四端点；证据不足时
  明确要求近拍，证据可用时按 3°/15°提供辅助分诊。
- 松动状态必须由操作员逐点确认。当前模型没有独立真实状态测试集证据，不能作为全自动松动判定器。
- 支持手机拍照、眼镜照片同步和原图库检测；显式设置 `CRRC_DETECTOR_BACKEND=onnx` 可构建旧的
  640 ONNX 候选后端作为回退。

### 万用表读数识别
- 基于通义千问视觉模型（`qwen3-vl-flash`）云端识别万用表读数与挡位。
- 支持「万用表拍照」语音触发，以及原图库内对任意照片执行识别。
- 识别结果写入**巡检台账**（本地记录），支持阈值报警与语音播报。

### 设备状态
- **实时电量**：周期查询眼镜电量（协议 `action=66`），在设备状态卡片实时显示电量与充电状态。

## 技术栈

- 原生 Android（Java），minSdk 22 / targetSdk 33
- BLE（Bluetooth Low Energy）连接与控制
- WiFi Direct（Wi-Fi P2P）+ HTTP 明文文件传输
- EventBus（`org.greenrobot.eventbus`）实现服务与 UI 通信
- ML Kit Barcode Scanning 二维码识别
- ONNX Runtime 离线目标检测（YOLO，NNAPI 加速）
- 通义千问视觉模型云端识别（万用表读数）
- 讯飞 SparkChain SDK（在线语音听写 ASR）
- 系统 TTS + ToneGenerator 语音播报与提示音
- K900 SDK（`ksdk-release.aar`）

## 项目结构

```
ar_glass_app/
├── app/
│   ├── src/main/
│   │   ├── java/com/ar/glass/
│   │   │   ├── core/                         # 核心逻辑
│   │   │   │   ├── GlassBleService.java      # BLE + WiFi Direct + HTTP 同步核心服务
│   │   │   │   ├── AppState.java             # 全局状态（眼镜 IP、电量等）
│   │   │   │   └── App.java                  # Application
│   │   │   ├── ui/                           # 界面
│   │   │   │   ├── MainActivity.java         # 主界面：连接、同步、语音入口
│   │   │   │   ├── GalleryActivity.java      # 原图库（网格浏览）
│   │   │   │   ├── ImageViewerActivity.java  # 全屏看图（长按三项检测）
│   │   │   │   └── MeterRecordsActivity.java # 巡检台账
│   │   │   ├── record/                       # 台账记录
│   │   │   │   ├── MeterRecord.java          # 台账数据模型
│   │   │   │   └── MeterRecordStore.java     # 台账本地存储
│   │   │   ├── util/EventMsg.java            # EventBus 事件定义
│   │   │   ├── voice/VoiceController.java    # 语音采集、识别与播报
│   │   │   └── vision/                       # 图像识别
│   │   │       ├── ImageAnalyzer.java        # 识别接口
│   │   │       ├── DefaultImageAnalyzer.java # ML Kit 二维码识别实现
│   │   │       ├── Vision.java               # 识别引擎入口
│   │   │       ├── YoloDetector.java         # YOLO 离线目标检测（ONNX）
│   │   │       ├── MeterReading.java         # 万用表读数模型
│   │   │       ├── ThresholdAlarm.java       # 阈值报警
│   │   │       └── cloud/MeterCloudOcr.java  # 通义千问云端识别
│   │   ├── res/                              # 布局、图标、字符串等资源
│   │   ├── jniLibs/                          # native 库（onnxruntime、sherpa-onnx 等）
│   │   ├── libs/                             # K900 SDK、SparkChain SDK 等 aar
│   │   └── AndroidManifest.xml
│   └── build.gradle
├── build.gradle
├── settings.gradle
└── gradle/                                   # Gradle Wrapper
```

## 构建与运行

模型文件使用 Git LFS 管理。首次克隆后先拉取完整模型，再用 Android Studio 打开项目：

```bash
git lfs install
git lfs pull
```

1. 确认 `app/src/main/ncnnAssets/model.ncnn.bin`、`app/src/main/ncnnJniLibs/`、
   `app/src/main/assets/marked-point-verifier.onnx`、`app/src/main/assets/witness-roi.onnx` 和
   `app/src/main/onnxAssets/fastener-target-p2-640.onnx` 已由 Git LFS 拉取完成。
2. 用 Android Studio 打开 `ar_glass_app` 目录，等待 Gradle 同步完成。
3. 连接 Android 手机（Android 5.1 / API 22 及以上），点击 Run 编译安装。

命令行构建：

```bash
./gradlew assembleDebug
```

生成的 APK 位于 `app/build/outputs/apk/debug/`。

默认构建无需设置 NCNN 模型或 JNI 目录环境变量，直接生成 NCNN + verifier + witness 三段链 APK。
构建会校验全部模型和预编译 JNI 的文件大小与 SHA-256；文件缺失、仍是 Git LFS pointer 或内容不匹配
时会直接失败，不会生成缺少算法资源的 APK。

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

常用命令（`action`）：

| action | 含义 | 示例 payload |
|--------|------|--------------|
| 65 | 眼镜控制 | 拍照 `{2,1,1}`、进入照片导入 `{2,1,4,1}`、退出导入 `{2,1,9}` |
| 66 | 电量查询 | `{0,0}` |
| 69 | 心跳 | `{4,1}` |
| 74 | 相机状态 | 开启相机 `{2,1,1}` |
| 115 | 数据上报 | 解析眼镜 IP（type=0x08） |

### 照片同步流程

1. 用户点击「同步照片到手机」。
2. 发送 BLE 命令 `glassesControl(new byte[]{2,1,4,1})`，让眼镜进入照片导入模式。
3. 建立 WiFi Direct 连接，通过 BLE `action=115`（type=0x08）数据帧解析出眼镜 IP。
4. HTTP 拉取照片列表：`http://<ip>/files/media.config`（兜底 `http://<ip>:80/storage/sd0/C/DCIM/1/vf_list.txt`）。
5. 把全部照片下载到临时目录 `glass_media/tmp`，弹窗显示缩略图供勾选。
6. 用户勾选后，选中照片移动到 `glass_media/photos`，未选中的从临时目录删除。
7. 发送 `glassesControl(new byte[]{2,1,9})` 退出导入模式（否则眼镜无法继续拍照）。

### 语音控制与拍照流程

1. 按住「按住说话」按钮，进入蓝牙 SCO 通话模式，通过 `AudioRecord` 采集眼镜麦克风（16k/16bit/mono PCM）。
2. 松手后退出 SCO，把 PCM 交给讯飞 SparkChain 在线语音听写识别。
3. 识别文本包含「拍照 / 拍摄 / 拍一张」时，触发拍照：先发 `action=74 {2,1,1}` 开启相机，延时 600ms 后再发 `action=65 {2,1,1}` 拍照。
4. 拍照完成后自动同步最新照片，并按语音意图分流——「二维码拍照」→ 二维码识别、「对齐拍照」→ YOLO 检测、「万用表拍照」→ 读数识别，结果通过 TTS 走媒体流（A2DP）从眼镜扬声器播报。

### 电量查询

- 连接成功后及每个心跳周期，发送 `action=66 {0,0}` 查询电量。
- 响应帧 payload `[0]` 为电量百分比，`[1]` 为充电状态（1 表示充电中），更新到设备状态卡片。

### 二维码识别

位于 `com.ar.glass.vision` 包，基于 ML Kit Barcode Scanning，采用四级递进识别：

1. 原图
2. 对比度增强
3. 放大 2 倍
4. 放大 2 倍 + 对比度增强

仅识别 `QR_CODE` 格式，限制最长边 2000px 以兼顾速度与识别率。

## 依赖

- `androidx.appcompat` / `com.google.android.material` / `androidx.constraintlayout`
- `org.greenrobot:eventbus:3.3.1`
- `com.squareup.okhttp3:okhttp:3.14.9`
- `com.google.mlkit:barcode-scanning:17.2.0`
- `io.netty:netty-all:4.1.68.Final`
- `com.google.protobuf:protobuf-javalite:3.18.1`
- `org.slf4j:slf4j-api` / `slf4j-simple`
- `com.google.code.gson:gson:2.8.8`
- `ksdk-release.aar`（K900 SDK）
- `SparkChain.aar` / `Codec.aar`（讯飞 SparkChain 在线语音听写）
- `com.microsoft.onnxruntime:onnxruntime-android`（YOLO 离线推理）

## 注意事项

- 语音识别走讯飞 SparkChain 云端，**手机需联网**（数据流量或 WiFi 均可）。
- 眼镜端 `media.config` 只列出「未传输」照片。照片一旦导入，眼镜会将其标记为已传输，之后无法再通过 `media.config` 列出（属眼镜固件行为，App 端无法改变）。
- 应用使用明文 HTTP 传输照片，已在 Manifest 中开启 `android:usesCleartextTraffic="true"`。
- 同步结束后必须发送退出导入模式命令 `glassesControl(new byte[]{2,1,9})`，否则眼镜会停留在占用状态、无法拍照。
- 提示音与播报走媒体流（A2DP），进入 SCO 通话模式时媒体流会被系统抑制，因此录音提示音在进入 SCO 前、退出 SCO 后播放。
