# AI 模型调用模块化 + ASR 上下文修正 调研与设计

> 日期：2026-09-15 ｜ 关联：TODO T12(模块化) / T13(ASR 上下文修正)
> 前置结论（真机已验证）：LLM/VLM/ASR 三条链路在 MuMu 16416 与真机 Xiaomi 25079RPDCC 均正常，
> 真机 `accel_dev=yes (Vulkan0)`、`CLIP using Vulkan0`、`vision prefill done in 30.1s`。

---

## 1. 现状：模型调用的耦合点（勘察结果）

| 位置 | 硬编码内容 | 影响 |
|---|---|---|
| `LlmEngine.java:19-34` | `VLM_ASSET` / `MODEL_ASSET` / `TEXT_ASSET` / `MMPROJ_ASSET`、`DEFAULT_CTX=4096`、`DEFAULT_GPU_LAYERS=99`、线程数 `4` | 换模型要改代码+重编译 |
| `LlmEngine.generate*` | ChatML 模板、`<think>\n\n</think>` 预填、媒体标记 `<__media__>`、温度/最大 token | 换模型家族（非 ChatML）要改代码 |
| `LlmEngine.ensureModelFile` | 分片命名 `%s.%02d.part`、<2GB 分片拼接逻辑 | 与具体资产名耦合 |
| `AsrEngine.java:24-27` | `asr/sensevoice-small-int8.onnx`、`asr/tokens.txt`、`SAMPLE_RATE=16000` | 换 ASR 模型要改代码 |
| `AsrEngine.load` | `language="zh"`、`use_itn=true`、`numThreads=2`、`provider=cpu`、`decoding=greedy_search`、`modelType=sense_voice_ctc` | 参数散落、无热词/纠错入口 |
| `AiPipeline.java` | `promptCorrect` / `promptSummarize` 模板、`MAX_CHARS=2500` | Prompt 与逻辑混写 |
| `AiChatActivity.java:77-79,191,237-238` | `new LlmEngine()` / `new AsrEngine()`、`ensureModelFile` / `ensureAssetFile(VLM/MMPROJ)` | 编排层直接依赖具体类与资产常量 |

**结论**：换一个模型/换一个 prompt，需要改 3~4 个文件并重编译。这是本次模块化要消除的。

---

## 2. 模块化设计

### 2.1 目标
**换模型 = 改一份 `models.json` / 放一个模型文件；换 prompt = 改模板文件。** 不改 Java 代码、不重编译业务逻辑。

### 2.2 新增结构（`com.ar.glass.ai.model` 包）

```
ai/model/
  ModelSpec.java        # 单个模型的声明式规格（POJO，可 JSON 反序列化）
  ModelRegistry.java    # 加载 models.json + 用户覆盖，提供 getActive(type) / resolveFile(spec)
  AssetInstaller.java   # 通用 assets→modelBaseDir 安装（含 .NN.part 分片拼接、幂等、进度回调）
  ChatTemplate.java     # 按 spec.chatTemplate 生成 prompt（chatml / chatml-think / minicpmv / plain）
assets/ai/models.json   # 模型注册表（可被 files/ai/models.json 覆盖）
```

### 2.3 `models.json` 示例（等价于当前行为）

```json
{
  "version": 1,
  "active": {
    "llm": "vlm-minicpm-v4_6",
    "mmproj": "mmproj-minicpm-v4_6",
    "asr": "sensevoice-small-int8"
  },
  "models": {
    "vlm-minicpm-v4_6": {
      "type": "vlm", "latencyClass": "offline",
      "asset": "vlm/minicpm-v4_6-q4_k_m.gguf", "dir": "vlm",
      "nCtx": 4096, "nGpuLayers": 99, "nThreads": 4,
      "chatTemplate": "minicpmv", "mediaMarker": "<__media__>",
      "defaultMaxTokens": 1024, "defaultTemperature": 0.7
    },
    "mmproj-minicpm-v4_6": {
      "type": "mmproj", "asset": "vlm/mmproj-f16.gguf", "dir": "vlm"
    },
    "llm-minicpm5-2b": {
      "type": "llm", "latencyClass": "offline",
      "asset": "llm/minicpm5-2b-q4_k_m.gguf", "dir": "llm",
      "nCtx": 4096, "nGpuLayers": 99, "nThreads": 4,
      "chatTemplate": "chatml-think"
    },
    "sensevoice-small-int8": {
      "type": "asr", "latencyClass": "realtime",
      "model": "asr/sensevoice-small-int8.onnx", "tokens": "asr/tokens.txt",
      "language": "zh", "useItn": true, "numThreads": 2,
      "provider": "cpu", "decoding": "greedy_search",
      "hotwords": { "file": "ai/hotwords.sherpa.txt", "score": 1.5, "enabled": false },
      "hr": { "lexicon": "ai/hr/lexicon.txt", "ruleFsts": "ai/hr/replace.fst", "dictDir": "ai/hr" }
    }
  }
}
```

### 2.4 引擎改造（**行为保持不变**）
- `LlmEngine`：常量改为 `ModelSpec` 字段；`load()` 接收 spec 的 `nCtx/nGpuLayers/nThreads`；prompt 由 `ChatTemplate.build(spec, userText, hasImage)` 生成。默认值 = 现有值，**输出逐字节不变**。
- `AsrEngine`：`load()` 读 spec（language/itn/threads/provider/decoding）+ 新增可选 `hr`（见 §3.3）。
- `AiChatActivity`：改为 `registry.getActive("llm")` 等，不再直接引用资产常量。
- 兼容层：保留原常量名并指向 registry 默认项，避免其它调用点（`AiHttpServer` 等）一次性全改。

### 2.5 分阶段计划
| 阶段 | 内容 | 风险 |
|---|---|---|
| P1 | 新增 `ModelSpec/ModelRegistry/AssetInstaller/ChatTemplate` + `models.json`；`LlmEngine/AsrEngine` 读 spec，默认值等价现状 | 低（行为不变） |
| P2 | `AiChatActivity`/`AiHttpServer` 改走 registry；prompt 外置 | 低 |
| P3 | 支持运行时切换（UI 下拉 / 覆盖 `files/ai/models.json`）+ 热重载 | 中 |
| P4 | 接入 ASR 纠错层（hr / 标点 / LLM 修正开关） | 中 |

**验收**：P1~P2 完成后，reload 日志与 `vision init ok` / prefill 耗时与改造前一致；把 `models.json` 的 `active.llm` 指向 `llm-minicpm5-2b` 能纯文本对话（预期视觉会失败——因维度不匹配，属已知约束，正好验证"换模型不改代码"）。

---

## 3. ASR"整体修正"调研

### 3.1 结论：SenseVoice 自带什么 / 不带什么
**自带**（模型能力，官方 README/论文）：
- ASR + 语种识别(LID) + 情感(SER) + 音频事件(AED)，**非自回归**，单次前向输出"整句"
- **ITN（逆文本归一化）**：标点 + 数字/日期规整（"三点五"→"3.5"、"百分之二十"→"20%"）——本项目已启用（`setInverseTextNormalization(true)`）

**不带**：
- ❌ 无"按对话上下文纠错"机制；❌ 无热词/上下文偏置；❌ 无 N-best 重打分；❌ 无纠错 LM
- 它不生成多候选，也不参考历史 utterance

> ⚠️ 澄清：本项目里"根据上下文修正内容"的**整体修正**，实现位置是 `AiPipeline.correct()` + `promptCorrect()`
> （用本地 LLM 做断句/标点/同音字纠错/删广告），**不是 SenseVoice 自带**。UI 上的「修正」按钮即此。

### 3.2 可用的三层纠错（现状 + 可加）

| 层 | 机制 | 状态 | 上下文感知 | 成本 |
|---|---|---|---|---|
| L1 ITN/标点 | SenseVoice 内置 | ✅ 已启用 | ❌ | 0 |
| L2 同音字/术语替换 | sherpa-onnx **Homophone Replacer**(hr)：lexicon + OpenFst `replace.fst` | ➕ 可加（接口已暴露） | ⚠️ 词典/规则级（非句义） | 低（无模型） |
| L2' 标点模型 | sherpa-onnx `OfflinePunctuation`(CNN-BiLSTM) | ➕ 可加 | ❌ | 低（~几十 MB） |
| L3 LLM 后处理 | `AiPipeline.correct`（本地 MiniCPM） | ✅ 已实现 | ✅ 真·上下文 | 高（一次 LLM 调用） |

### 3.3 实现路径 A：sherpa-onnx 同音字替换（推荐先做 L2）
**官方已支持 SenseVoice + hr**（sherpa-onnx PR #2153 / `sense-voice-with-hr-cxx-api.cc`）：
```
--sense-voice-model=model.int8.onnx --sense-voice-use-itn=1
--hr-lexicon=lexicon.txt --hr-dict-dir=dict --hr-rule-fsts=replace.fst
```
需要三样东西（需自建，针对本项目术语）：
1. `lexicon.txt`：词 → 音素序列（发音词典）
2. `dict/`：由词典派生的中间目录
3. `replace.fst`：OpenFst 规则，形如 `错误写法 → 正确写法`

**FST 生成**：官方给了 colab（见 PR #2153）；离线也可用 `pynini` / `kaldifst` 自建脚本生成。

**代码改动点**（很小）：
```java
// AsrEngine.load(...) 增加：
OfflineRecognizerConfig.builder()
    .setHr(HomophoneReplacerConfig.builder()
        .setLexicon(spec.hrLexicon)      // files/asr/hr/lexicon.txt
        .setRuleFsts(spec.hrRuleFsts)    // files/asr/hr/replace.fst
        .setDictDir(spec.hrDictDir)
        .build())
```
> 注意：`hr.dictDir` 在当前 Java 层注释为 unused，但 C++ 侧会读取；需与 `ruleFsts` 路径一并提供。

**适用场景**：项目专有名词（"紧固件""防松""工务""点检""互译""机种"等）被误识别的确定性修正，零延迟、零算力。

### 3.4 实现路径 B：LLM 上下文修正（已有，建议增强）
现状：`promptCorrect` 已要求"根据上下文推测并修正同音字"。可增强项：
- 传入**最近 N 轮对话上下文**（目前是单轮），让 LLM 有真正的"上下文"依据
- 传入**领域词表**（把 L2 的词表作为 prompt 提示），提高术语命中
- 流式场景：分段上屏用轻量规则，**松手终稿**再走 LLM 修正（已是当前架构）

### 3.5 实现路径 C：独立标点模型（可选）
`sherpa-onnx OfflinePunctuation`（CT-Transformer CNN-BiLSTM）：若某些模型（如未来的流式 ASR）不带标点，可挂它补标点。SenseVoice 已带标点，**暂不需要**。

### 3.6 建议
1. **先做 L2（hr 同音字替换）**：确定性强、零算力，先把专有名词错误率压下来；需要你提供/确认**领域术语表**。
2. **再做 L3 增强**：把"对话上下文 + 领域词表"喂给 LLM 修正，逼近"按上下文整体修正"的直觉。
3. L1 已启用，保持。
4. hr 与 LLM 修正**不冲突**：hr 先跑（确定性替换），LLM 再修（语义级）。

---

## 4. 热词字典与语音唤醒（预留）

### 4.1 已落地
- **字典文件**：`app/src/main/assets/ai/hotwords.txt`（内置默认；放 `src/main/assets` 故**精简包也保留**）
  - `[wake]` 段 = 唤醒词；`[term]` 段 = 领域术语；支持 `词 :权重`；`#` 注释；空行忽略
  - **设备侧覆盖**：`/storage/emulated/0/Android/data/com.ar.glass/files/ai/hotwords.txt`（改完即生效，免重编译）
- **读取器**：`com.ar.glass.ai.model.HotwordStore`
  - `wakeWords()` / `terms()` / `allWords()` / `matchWake(text)`（文本侧唤醒命中）
  - `exportSherpaFile(ctx)` → 生成 `files/ai/hotwords.sherpa.txt`（sherpa 兼容热词表，每行一条）
- **ASR 接线（新增，默认关闭、行为不变）**：`AsrEngine.Options`
  - `hrLexicon / hrRuleFsts / hrDictDir` → 同音字替换
  - `hotwordsFile / hotwordsScore` → 热词（见下，当前栈不消费）
  - `load(ctx)` 自动探测 hr：**仅当 lexicon 与 replace.fst 同时就位才启用**，缺一则关闭
- **hr 占位**：`app/src/main/assets/ai/hr/lexicon.txt`（含生成说明）。`replace.fst` 需按 §3.3 生成后放入同目录

### 4.2 关键事实（避免走弯路）
| 机制 | SenseVoice (CTC) | Transducer / Paraformer |
|---|---|---|
| `hotwordsFile` 解码偏置 | ❌ 不消费 | ✅ 支持 |
| `hr` 同音字替换 | ✅ 官方支持（PR #2153） | ✅ 支持 |
| ITN（标点/数字日期） | ✅ 已启用 | 视模型 |

**结论**：当前栈（SenseVoice）要让热词真正"起作用"，途径是 **hr——把术语写进 lexicon + replace.fst**；
`hotwordsFile` 仅作**预留**，等将来换 Transducer/Paraformer 模型时启用。

### 4.3 语音唤醒的后续路线（预留，未实现）
1. **KWS 专用模型（推荐，功耗优先）**：sherpa-onnx `KeywordSpotter`——项目已 vendored
   `KeywordSpotter.java / KeywordSpotterConfig.java / KeywordSpotterResult.java`（与 `libsherpa-onnx-jni.so` 同源）。
   常驻低功耗监听，命中唤醒词再开 ASR；需接入 KWS 模型（如 `sherpa-onnx-kws-zipformer-*`）。
2. **文本侧唤醒（最省事，快速验证）**：复用流式 ASR + `HotwordStore.matchWake(text)`——零新模型，但 ASR 需常开、功耗较高。
3. **配合 hr**：若唤醒词本身常被误识别，先经 hr 纠回再匹配。

---

## 5. 风险 / 待确认
- P1 改造需保证**行为逐字节不变**，用真机日志（reload + vision init + prefill 耗时）做回归基线。
- `hr.dictDir`/`ruleFsts` 的路径需在设备上真实存在，否则 sherpa-onnx 可能启动失败 → `AssetInstaller` 需支持目录级安装。
- `replace.fst` 生成链路需要在 PC 端搭（pynini/kaldifst），确认是否接受引入新工具链。
- **待确认**：模块化是走"JSON 配置注册表"（推荐）还是"UI 内下拉切换"？见对话中的选项。
