# Feature_Android_AI 分支说明

安卓本地 AI 分支：SenseVoiceSmall（ASR）+ Qwen3-4B（LLM）全部离线运行，
不依赖网络。完整包含 AR-glasses-demo master 的全部内容，新增 `com.ar.glass.ai` 后端模块。

## 架构

| 组件 | 引擎 | GPU | 模型资产 |
|---|---|---|---|
| LLM | llama.cpp（vendor 在 `app/src/main/cpp/ai/llama.cpp/`），JNI `libai_llm.so` | Vulkan（显存不足自动 CPU 分层回退） | `assets/llm/qwen3-4b-q4_k_m.gguf`（2.5GB，noCompress） |
| ASR | sherpa-onnx JNI（`libsherpa-onnx-jni.so`，master 源码自编译，shared ORT） | CPU int8（ONNX Runtime） | `assets/asr/sensevoice-small-int8.onnx`（240MB）+ `tokens.txt` |
| 管线 | `AiPipeline`：转写→修正→摘要（prompt 移植自 PC 端 bilibili.bat） | — | — |

- 前端仅一个简易页：`AiChatActivity`（文本对话 + 按住说话 + 修正/摘要按钮），入口在主界面"AI 助手"按钮。
- `ThinkFilter`：qwen3 `<think>` 思考区流式剥离（借鉴 F:\Chat2API 的跨分片状态机设计）。
- abiFilters 收敛为 arm64-v8a（qwen3:4b 运行需约 3.5GB 可用内存）。

## 大模型资产不在 git 内（体积超限），按下面步骤恢复

1. **LLM GGUF**（来自本机 Ollama qwen3:4b 缓存）：
   ```powershell
   copy "C:\Users\29268\.ollama\models\blobs\sha256-3e4cb14174460404e7a233e531675303b2fbf7749c02f91864fe311ab6344e4f" `
        "app\src\main\assets\llm\qwen3-4b-q4_k_m.gguf"
   ```
2. **SenseVoice ONNX**：已导出产物存于 `F:\Android_YOLO\Temp\sherpa-onnx-clone\scripts\sense-voice\`
   （`model.int8.onnx` + `tokens.txt`），或按第 3 步重新导出；复制到 `app/src/main/assets/asr/`。
3. **重新导出（可选）**：`model-cache/`（gitignore）内是 modelscope 原始缓存克隆
   （SenseVoiceSmall + fsmn-vad）。导出脚本在 `F:\Android_YOLO\Temp\sherpa-onnx-clone\scripts\sense-voice\export-onnx.py`，
   需 `D:\download\bilibili\.venv`（funasr + onnx + onnxruntime + protobuf）。
   注意：官方 `model.py` 的 `SinusoidalPositionEncoder.__init__` 缺 `super().__init__()`，
   新版 torch 会报 `_state_dict_pre_hooks` 错误，需先修补。

## 已知约束

- APK 约 3GB，adb 侧载安装。
- `aaptOptions noCompress 'gguf','onnx'`，保证 GGUF 可直接 mmap（`LlmEngine.ensureModelFile`
  首启动会把 assets 拷到 filesDir）。
- llama.cpp 编译需要 NDK shader-tools 的 glslc 与 SPIRV-Headers（CMakeLists 内已写死本机回退路径，
  换机需调整 `app/src/main/cpp/ai/CMakeLists.txt`）。
- sherpa-onnx JNI 为 shared-ORT 构建，需同时打包 `libsherpa-onnx-jni.so` 与 `libonnxruntime.so`；
  build.gradle 的 `packagingOptions pickFirst` 已处理与 onnxruntime AAR 的重复。
