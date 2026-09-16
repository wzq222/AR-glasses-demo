# Feature_Android_AI 分支说明

安卓本地 AI 分支：SenseVoiceSmall（ASR）+ MiniCPM5-2B（文本 LLM）+ MiniCPM-V 4.6（多模态视觉）
全部离线运行，不依赖网络。完整包含 AR-glasses-demo master 的全部内容，新增 `com.ar.glass.ai` 后端模块。

## 架构

| 组件 | 引擎 | GPU | 模型资产 |
|---|---|---|---|
| 文本 LLM | llama.cpp（vendor 在 `app/src/main/cpp/ai/llama.cpp/`），JNI `libai_llm.so` | Vulkan（Mali 自动禁 coopmat，见下） | `assets/llm/minicpm5-2b-q4_k_m.gguf`（1.6GB，noCompress） |
| 多模态视觉 | llama.cpp mtmd（`tools-mtmd/`，MiniCPM-V 4.6 已合入官方支持） | Vulkan（mmproj use_gpu） | `assets/vlm/minicpm-v4_6-q4_k_m.gguf`（529MB）+ `mmproj-f16.gguf`（1.11GB） |
| ASR | sherpa-onnx JNI（`libsherpa-onnx-jni.so`，master 源码自编译，shared ORT） | CPU int8（ONNX Runtime） | `assets/asr/sensevoice-small-int8.onnx`（240MB）+ `tokens.txt` |
| 管线 | `AiPipeline`：转写→修正→摘要（prompt 移植自 PC 端 bilibili.bat） | — | — |

- 文本模板：MiniCPM5 ChatML + 预填空 `<think>` 块跳过混合思考（快速响应）。
- ASR 流式：按住说话期间每 0.8s 分段转写实时上屏（SenseVoice 离线模型分段伪流式）；
  RTF>1.5 自动降级整段识别并在 `AiLog` 说明"设备算力不足"；松手后全量重转写为终稿。
- 本地 API：`AiHttpServer`（NanoHTTPD），`POST /v1/chat/completions`（stream:true 走
  JNI `nativeGenerateStreaming` 逐 token 回调 → Piped 流 → ThinkFilter → 真·SSE）、
  `GET /v1/models`；Bearer API_KEY（`sk-local-xxx`，存 SharedPreferences，AI 页显示）；
  端口 8080；云端/本地切换只需改客户端 base_url。
- 图片输入：`LlmEngine.generateWithImage(Bitmap, prompt)` → JNI `mtmd_tokenize`（媒体标记
  `<__image__>`）→ `mtmd_helper_eval_chunks` 视觉预填充 → 常规解码。前端"图片"按钮选图（≤896px）；
  调试广播 `AiDebugReceiver`（`--es cmd img --es path`）可注入测试图绕过选择器。
- `ThinkFilter`：`<think>` 思考区流式剥离（借鉴 F:\Chat2API 的跨分片状态机设计）。
- GPU 判定计入 IGPU（ARM UMA 的 Mali 注册为集成 GPU 类型，只查 GPU 会误报 CPU）。
- abiFilters：arm64-v8a（真机）+ x86_64（模拟器）；ai_llm 的 native 平台为 android-28（Vulkan 符号）。
- 已知未决：mmproj `vision init FAILED` 根因未定（mtmd/clip 日志器独立于 ggml、默认写 stderr，
  已挂 `mtmd_log_set` 转发 logcat，重测可拿具体报错）。

## 已知 GPU 兼容性

- Mali/Immortalis（天玑）驱动 coopmat 有 bug：检测到 Mali 自动设 `GGML_VK_DISABLE_COOPMAT=1`
  （`LlmEngine.initGpuCompat`，真机 G925 已验证 GPU 全量驻留：权重 2.4GB + 36 层 KV + 计算图）。
- 仍回退 CPU 时自动升级 tier2（`GGML_VK_DISABLE_F16=1`），持久化后重启生效。

## 构建

- 默认完整包（约 3.6GB，模型内置）；`CRRC_AI_BUNDLE_MODELS=0` 出精简包（~130MB，
  模型走 AI 页面路径接口外置，需授予"所有文件访问"）。
- ASR：按住说话即流式识别（0.8s 分段），无需额外资产；SenseVoice 解包在
  `Android/data/com.ar.glass/files/asr/`。
- 大模型资产不在 git（.gitignore），恢复方式：
  - LLM：`ollama pull openbmb/minicpm5-2b:q4_K_M`，从
    `~/.ollama/models/blobs/sha256-bfb18f3b...` 复制为 `assets/llm/minicpm5-2b-q4_k_m.gguf`
  - VLM：`hf-mirror.com/openbmb/MiniCPM-V-4.6-gguf` 下载 `MiniCPM-V-4_6-Q4_K_M.gguf` →
    `assets/vlm/minicpm-v4_6-q4_k_m.gguf`、`mmproj-model-f16.gguf` → `assets/vlm/mmproj-f16.gguf`
- llama.cpp 编译需 NDK shader-tools glslc + vendored SPIRV-Headers/Vulkan-Headers +
  `prebuilt/vulkan-shaders-gen.exe`（MinGW fork() 死锁，host 工具强制 MSVC toolchain）。
