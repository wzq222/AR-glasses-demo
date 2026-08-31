# 防松标记检测准确率恢复决策

## 结论

不要继续轮换YOLO-S/M，也不要把DINOv3塞进手机。推荐保留YOLOv8s-P2作为第一阶段proposal，改为
专门检测`marked_point`，配合切片训练和困难负样本；第二阶段用MobileNetV3对原图ROI做
`有效防松检查点 / 无标记紧固件 / 外观相似物 / 无法判断`复核。最终阈值按高召回选，疑难项交给人。

## 为什么这样改

本地误差显示，换大模型不能解决问题：YOLOv8m-P2比S-P2更差；主要损失来自tiny与lookalike。
在marked-point开发集中，fastener来源的726个候选已覆盖248/248真值，而color-only产生8,841个候选。
因此颜色分支不应继续作为独立全图候选源，而应在ROI里辅助判断是否有跨部件红/黄标记。

切片训练/推理能让微小目标占据更多输入像素；SAHI论文在多个检测器上报告切片推理与切片微调的AP
增益，但这类增益来自特定数据集，项目仍需自己的场景隔离验证。[1][7]
Focal Loss与OHEM的共同结论是：大量简单背景会淹没真正困难的负例，训练必须聚焦难例，而不是无上限
堆背景。[2][3]

## 固定训练路线

1. **YOLO-P2 proposal**：从旧P2权重初始化，类别改为单一`marked_point`。训练输入同时包含原全图和
   2×2重叠切片；普通无标记螺栓、红色热缩管、锈迹、标签和管线全部作为背景。
2. **困难负样本闭环**：每轮只加入模型最高分FP和最低分TP附近的hard negatives，按scene去重；不把
   8,841个颜色块平均灌入训练。
3. **MobileNetV3 ROI复核**：使用原图扩大约1.6倍的候选裁剪，输入224或256；MobileNetV3本身是面向
   手机CPU设计的硬件感知网络，适合承担第二阶段分类。[4]
4. **候选融合**：运行时采用YOLO全图低阈值候选，加少量颜色触发的高分辨率ROI补检；颜色不再产生
   数千个独立最终框。
5. **人工确认**：复核器输出概率和1×/2×/4×证据；低置信度、标记归属不明或状态疑似时交给人。

## 三组有界实验

| 实验 | 改动 | 目的 |
|---|---|---|
| E1 | marked-point单类YOLO-P2，原图+切片 | 验证目标口径修正和tiny召回 |
| E2 | E1 + hard-negative mining | 拉开lookalike置信度 |
| E3 | E2 proposal + MobileNetV3 verifier | 在保持召回时提高precision |

固定三个seed。proposal门：真实val recall不低于0.99，并记录每图候选数；最终门：recall不低于0.95、
precision不低于0.90、完整场景率不低于0.90。两轮提升小于1个百分点就停止调参，转为补真实场景。

## DINOv3的位置

DINOv3提供强大的冻结特征和轻量适配能力，但其ViT patch size为16，且官方模型有独立许可与权重访问
条件；它不适合作为本项目第一版手机运行模型。[5][6]
如果已合法取得权重，可离线用于聚类9,000余个lookalike、挑选多样hard negatives或做教师特征；不能用
它在同源小验证集上自证生产准确率。

## 限制

- 当前真实train只有30个独立场景、173个marked-point框，能训练研究模型，但不足以证明跨车辆可靠。
- 本路线提高的是“防松检查点在哪里”；“松不松”仍需要真实受控状态对和端点/部件标注。
- 任何达到门限的同源val结果都必须再经新增跨设备封存测试，才能接入Android。

## 非显而易见的关键点

当前最值得做的不是升级YOLO版本，而是把一个困难的“全图一次性高精度检测”拆成“高召回proposal +
局部高精度复核”。这与现有数据形态最匹配，也最容易在手机上控制算力。

## References

[1] Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection. https://arxiv.org/abs/2202.06934

[2] Focal Loss for Dense Object Detection. https://arxiv.org/abs/1708.02002

[3] Training Region-based Object Detectors with Online Hard Example Mining. https://arxiv.org/abs/1604.03540

[4] Searching for MobileNetV3. https://arxiv.org/abs/1905.02244

[5] Meta AI: DINOv3. https://ai.meta.com/research/dinov3/

[6] DINOv3 Model Card. https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md

[7] Ultralytics SAHI tiled inference guide. https://docs.ultralytics.com/guides/sahi-tiled-inference

