# AI 模型恢复指南（model/asr-llm 分支）

> 适用：需要从零恢复 Android AI 分支所需的全部大模型权重。
> 模型权重体积巨大（原始 ~3.49 GB），**不放在代码分支**，而是单独存放在
> **`model/asr-llm`** 分支，以 `tar.gz` 分卷（每卷 45 MB）形式提交。

---

## 1. 为什么单独开分支 + 分卷

- GitHub 单文件 **50 MB 硬上限**（GH001），直接提交原始 `.gguf`/`.onnx` 会被拒。
- 本机无 7-Zip，采用 **`tar -czf - | split -b 45m`**：每卷 < 50 MB，普通 `git` 推送即可，
  且慢网下推送按**对象级续传**（失败重推只补缺失卷）。
- 该分支为 **orphan（根提交仅含 `models/` 目录）**，与 `Feature_Android_AI` 代码分支无共同历史，互不干扰。

压缩后总体积约 **2.93 GB**，拆成 **67 个分卷**（前 66 卷各 45 MB，末卷 15 MB）。

---

## 2. 模型清单

| 模型 | 解压后仓库内路径 | 用途 | 原始大小 |
|---|---|---|---|
| MiniCPM5-2B (q4_k_m) | `app/src/aiAssets/assets/llm/minicpm5-2b-q4_k_m.gguf` | 文本 LLM（llama.cpp / Vulkan） | 1.49 GB |
| mmproj-f16 | `app/src/aiAssets/assets/vlm/mmproj-f16.gguf` | 多模态投影器（mtmd） | 1.03 GB |
| MiniCPM-V 4.6 (q4_k_m) | `app/src/aiAssets/assets/vlm/minicpm-v4_6-q4_k_m.gguf` | 视觉多模态 | 0.49 GB |
| SenseVoiceSmall int8 | `app/src/aiAssets/assets/asr/sensevoice-small-int8.onnx` | 流式 ASR（sherpa-onnx） | 0.22 GB |
| fastener-target-p2-640 | `app/src/main/assets/fastener-target-p2-640.onnx` **及仓库根目录** | YOLO 紧固件检测 | 42 MB |
| marked-point-verifier | 仓库根目录 `marked-point-verifier.onnx` | YOLO 标记点校验 | 5.9 MB |

> `fastener-target-p2-640.onnx` 在分支里存了两份：一份在 `app/src/main/assets/`（构建用副本），
> 一份在仓库根目录（源文件，`AGENTS.md` 规定根目录 onnx 为源、勿删勿改）。
> `marked-point-verifier.onnx` 只存于根目录。

---

## 3. 如何获取分卷

**方式 A（最干净，只拉模型分支）：**

```bash
git clone -b model/asr-llm --single-branch --depth 1 \
  https://github.com/wzq222/AR-glasses-demo.git models_tmp
cd models_tmp   # 内含 models/（67 个分卷 + MANIFEST.txt）
```

**方式 B（已在 Feature_Android_AI 仓库内，稀疏取模型目录）：**

```bash
git fetch origin model/asr-llm
git checkout origin/model/asr-llm -- models
```

---

## 4. 如何恢复（解压）

在**仓库根目录**执行（分卷按字母序 `cat` 后管道解压，无需中间文件）：

```bash
cat models/models.tar.gz.part-* | tar -xzf -
```

解压后模型落到上面的「仓库内路径」。其中：

- `app/src/aiAssets/assets/...` 与 `app/src/main/assets/fastener-target-p2-640.onnx`
  即 App 构建所需权重；
- 根目录的 `fastener-target-p2-640.onnx` 与 `marked-point-verifier.onnx`
  是 onnxruntime 管线的源文件，也应保留在 workspace 根目录 `F:/Android_YOLO/`。

---

## 5. 完整性校验

- **每卷 SHA256**：见 `models/MANIFEST.txt` 的 `## SHA256 of each volume` 段。
  下载/解压前可逐卷比对，任一卷不符即说明传输损坏。
- **每模型 SHA256**（原始权重，解压后可重算比对）：

  | 文件 | SHA256 |
  |---|---|
  | `minicpm5-2b-q4_k_m.gguf` | `bfb18f3bd98c3fe9d41bb057d7b564de14b5b7f143860e6e8e5f10ae5553005e` |
  | `mmproj-f16.gguf` | `ca931d861d0801d9003e50697cd764721a334107c0e0415a51168ee1938462de` |
  | `minicpm-v4_6-q4_k_m.gguf` | `6b0c74962c44bc6bf4b655b9b02c13eda9d5a0491543ae976d1ac18e4b7892e2` |
  | `sensevoice-small-int8.onnx` | `ac4d8740dafa562934441d9300533c0171adb023e242a6c687fc54fc401d8c91` |
  | `fastener-target-p2-640.onnx` | `c50f9105ff75885be3ba02464e6a994fa7a45fde0b0634aea12faa04a6cc5b7a` |
  | `marked-point-verifier.onnx` | `fed197a11134dd4358b70eff64086c050ddecc9b2c484e72aaeb102e4ba563cd` |

  校验命令示例：

  ```bash
  sha256sum app/src/aiAssets/assets/llm/minicpm5-2b-q4_k_m.gguf
  ```

---

## 6. 关于 model.pt（SenseVoice 导出源）

`model-cache/SenseVoiceSmall/model.pt`（约 936 MB）是 funasr 导出的**源模型**，本分支**未纳入**。
如需重新生成 `sensevoice-small-int8.onnx`，按 `docs/AI-BRANCH.md` 的 funasr 导出流程处理。

---

## 7. 注意事项

- 分卷必须**全部**下载，且按 `aa → co` 顺序（`models/models.tar.gz.part-*` 通配天然按字母序）。
  **任一卷缺失或损坏，整条 gzip 流都无法解压**。
- 分卷为普通 git blob（非 LFS），直接 `git pull` 即可，但 2.93 GB 首次拉取较慢。
- 重建后建议跑一次 `git lfs status`（代码分支）确认 `.so` 等 LFS 文件无异常；
  模型权重不归 LFS 管理。
- 该分支为 **orphan**，历史由 6 个根提交组成（仅含 `models/` 目录：1 个 MANIFEST + 5 个分卷块），
  与代码分支无共同历史，不要从它发起合并到代码分支。
