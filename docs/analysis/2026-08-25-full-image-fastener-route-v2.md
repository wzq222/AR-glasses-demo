# 全图防松线检测路线 V2（2026-08-25）

## 管理结论

停止把全图HSV候选当作训练标签。60.89%是颜色规则候选精度，不是模型性能；当前没有训练模型。
颜色规则的失败不是阈值未调好，而是锈蚀、红色结构、线缆、警示贴和反光与防松色标存在外观重叠。

主线改成对象优先的两级检测：

1. `PicoDet-S 416`先做全图上下文，再对2×2重叠切片做物理紧固件检测并合并结果；若目标手机余量足，再升`PicoDet-M`。
2. 仅在紧固件ROI内，用小型关键点热图头或OpenCV提取色标、两个端点与跨接关系。
3. 几何层输出`NORMAL / SUSPECTED_LOOSE / UNCERTAIN`；低置信度不自动判松动。
4. `RTMDet-tiny-P2`作为离线教师/精度对照，`D-FINE-N`降为服务器挑战者；`DINOv3`和
   `Grounded-SAM-2`只做离线标注、聚类或相邻帧传播，不进入APK主线。

## 为什么必须切片

151个已接受色标候选在2000×1500原图中的线段长度中位数为22 px。整图缩到416时中位仅4.6 px，
缩到640时仅7.0 px；约1100 px宽的重叠切片缩到640后约12.8 px。一次整图缩放会主动抹掉小目标信息。
“全图检测”应定义为完整覆盖，而不是强制一次神经网络前向。

## 标注与训练动作

- 从177个场景组选择80—120个代表帧，人工标注物理紧固件框；色标存在性、颜色、两个端点和是否跨接作为ROI属性。
- 把锈蚀、线缆、警示贴、反光和无色标紧固件作为困难负样本保留。
- 用Grounded-SAM-2、HSV候选和相邻帧传播产生建议，不直接生成真值；人工修正后训练首版PicoDet-S。
- 首版模型回扫未标数据，优先复核高不确定和高误报画面，迭代一轮后再冻结内部验证。
- 另采一批跨车辆/日期/光照的独立test，并拍摄同位置正常、轻微错位、明显错位和无色标的受控状态。

## 验收门

以下是项目目标，不是当前结果：

- 独立test紧固件召回不低于95%，平均误报不高于0.5个/图。
- 指定Android手机从Bitmap输入到结果输出的普通扫描P95不高于300 ms；低置信度精查P95不高于800 ms。
- 记录P50/P95、峰值内存、连续50次热机推理与温降频；PicoDet-S/M用同一数据和切片策略正面对比。
- 没有受控松动状态和独立test前，只能报告检测/关键点指标，不能报告松动识别准确率。

## 官方依据

- [PicoDet移动端规模、基准与Android部署](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/picodet/README_en.md)
- [PaddleDetection小目标与切片方案](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/smalldet/README.md)
- [RTMDet官方模型结果](https://github.com/open-mmlab/mmdetection/blob/main/configs/rtmdet/README.md)
- [MMDetection半自动标注闭环](https://github.com/open-mmlab/mmdetection/blob/main/docs/en/user_guides/label_studio.md)
- [Grounded-SAM-2自动标注与跟踪](https://github.com/IDEA-Research/Grounded-SAM-2)
- [DINOv3官方实现](https://github.com/facebookresearch/dinov3)

