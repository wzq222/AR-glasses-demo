# 防松线合成数据训练消融设计

## 目标

在不查看sealed-test、不修改formal truth的前提下，判断24张ImageGen防松线全图是否能提升
YOLOv8s-P2在真实val上的检查点检测召回。该试验只评价检测框，不把合成状态标签当作真实松动
准确率证据。

## 对照

- control与synthetic均从同一个固定权重开始，使用seed `20260829`、20 epoch、640输入、batch 4。
- control每epoch为39个batch；synthetic同样为39个batch，保持优化步数一致。
- synthetic每个batch固定3张真实视图和1张合成视图，合成占比25%，严格低于30%。
- synthetic训练COCO由真实train与24张已严格审核的合成全图合并；真实val保持原文件和原哈希。

## 数据与安全门

- 合并前重跑合成全图严格审计，验证formal truth、全图、COCO、裁剪图和审核包哈希。
- 合成图只允许进入train；图像ID从`1000000`开始，便于物化后识别来源。
- 混合COCO和所有训练产物只写入`E:/crrc_vision_data`，不提交现场图像或权重。
- 采样器按epoch和seed确定性重排，单个epoch的batch数与control一致。

## 决策门

只使用同一真实val比较precision、recall、mAP50和mAP50-95。优先目标是recall提升且precision不出现
不可接受下降；若合成臂没有稳定改善，保留数据流水线但不采用该权重。无论结果如何，都不宣称已具备
真实防松线松动状态准确率。
