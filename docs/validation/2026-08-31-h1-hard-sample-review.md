# H1 困难防松标记生成审核结论

## 结论

H1a 的 24 个固定任务共形成 41 条 ImageGen 尝试记录。经原尺寸检查、必要的隐藏结论二审和
SHA-256 绑定审核，18 个任务至少有一张 `APPROVED`，6 个任务在最多三次尝试后仍未通过。因此 H1a
固定配比门保持关闭，未构建 24 张全图训练集，也未把未通过图像写入训练真值。

未通过任务为 `h1a-0008`、`h1a-0009`、`h1a-0010`、`h1a-0019`、`h1a-0020`、`h1a-0024`。
第三轮 6 张的审核结果为 `0 APPROVED / 6 REJECTED / 0 UNCERTAIN`。

## 根因

1. `SUBTLE_DISPLACED / OBVIOUS_DISPLACED` 图像经常保持防松线共线，或只改变颜色纹理，没有产生
   可验证的 moving/fixed 相对位移；继续提示词重试不能建立物理真值。
2. `ref-12` 裁剪同时包含显著管接头和较小螺母，原 `nut_plate` 任务标签不能唯一绑定目标，导致
   `INSUFFICIENT / LOOKALIKE` 两个任务发生拓扑漂移。
3. 这些失败不是普通噪声增强问题。把图像视觉真实性当作机械状态真实性会向状态模型注入错误监督。

## 决策

- 停止上述六个任务的 ImageGen 重试，保留全部失败尝试和审核记录用于审计。
- 18 个已通过局部样本仍只作为研究资产；H1a 总门未通过前不得拼成 24 张固定配比训练集。
- 松动状态的生产阈值只使用真实受控 `ALIGNED / DISPLACED` 成对采集标定。ImageGen 可继续覆盖油污、
  反光、模糊、遮挡等外观困难，但不能单独证明机械相对位移。
- 现场流程转为人机协同：高召回发现防松标记检查点，回到原图裁剪，并提供 1×、2×、4×像素保真
  证据；人判为 `ALIGNED / POSSIBLE_DISPLACED / DISPLACED / DAMAGED_MARK / INSUFFICIENT`。
  放大图不使用生成式超分辨率，避免制造不存在的线段。

真实 `marked-point` 候选复核包也已加入同一证据结构：每个候选在原有带框 JPEG 上下文之外，新增
无标注 PNG 原像素裁剪、2× nearest 和 4× nearest 三个哈希绑定视图。该改动只增强人工观察，不改变
候选集合、不自动产生状态标签，也不修改既有真值。

## 验证证据

- 第三轮审核：`E:/crrc_vision_data/synthetic/marked-point-h1/h1a/reviews-attempt-03/reviewed-results.json`
- 1×/2×/4×证据包：`E:/crrc_vision_data/synthetic/marked-point-h1/h1a/review-pack-attempt-03-evidence-v2/manifest.json`
- Python：新增证据测试与全套测试均通过；最终运行结果见项目状态。
- 正式真值 SHA-256：
  `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`

本结论证明审核门能拒绝伪物理样本，不证明真实松动识别已经达到生产准确率。
