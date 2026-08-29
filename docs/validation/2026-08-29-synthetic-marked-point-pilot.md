# ImageGen 防松线合成试点验证

日期：2026-08-29

## 结论

试点达到既定的24张全图门，可进入低权重synthetic-training ablation；不能据此声称真实现场的
松动识别准确率或生产可用性。

## 数据与来源边界

- Git外资产根：`E:/crrc_vision_data/synthetic/marked-point-v1/`。
- 8个来源均来自train独立场景；val和sealed-test未用于生成、调参或审核。
- 最终局部集24张：NORMAL、SLIGHT_LOOSE、OBVIOUS_LOOSE各8张。
- 最终全图集24张，覆盖8个独立来源场景，三种状态各8张。
- 防松线涂料像素来自ImageGen生成图；程序只做掩膜提取、分段仿射重定位、有限透视和光度变换，
  不绘制红/黄线。
- 全图合成只迁移ImageGen涂料像素到对应来源场景的原检查点，不搬运整个紧固件外观；原有红/黄
  标记先经面积上限、中心邻域和颜色组件门做保守修复。大面积红漆、锈蚀误选、明显贴片边界和双线
  样本在试点中被淘汰并换用干净结构。
- mark-only模式严格保留真实检查点原COCO检测框；只变换防松线像素、两段端点和共享锚点。
- 背景COCO必须声明`train`分区，来源图和背景图均在合成前校验SHA-256。

## 双层复核

1. 逐张检查24张局部图，确认两段线落在同一检查点的固定面和活动面，角度为0°、8°、24°。
2. 逐张检查24张放大检查点，并按8个独立场景检查整图，确认没有方形贴片、异种紧固件替换、
   相邻目标覆盖或明显旧线残留。
3. 最终全图manifest为24/24 APPROVED，uncertain和rejected均为0。
4. 审核结论绑定全图SHA-256、裁剪图SHA-256和审核包manifest SHA-256；任一字节变化都会使
   严格全图门失败。

## 自动门与哈希

- 局部审计：24/24 APPROVED；三类各8；0错误。
- 全图审计：24/24 APPROVED；三类各8；0错误。
- Python：`248 passed`。
- 固定种子：`20260829`。
- 两次独立输出的内容哈希一致：
  `68C2A420E151329C9CF57D894EEE39243B8FC8CA26119F5953F27DD7BB540016`。
- `approved-locals.json` SHA-256：
  `FF666BA52427E2F5FE19963B92F706432307281216058FB78C547FC565562B8C`。
- 已审核全图`manifest.json` SHA-256：
  `5AF5A6EA4FFA2FD81EB2F75E055D5307E6298DEEED70F96B0C72FD8656FF48B3`。
- `instances.synthetic-train.json` SHA-256：
  `374D283327EEA6E906498E043A705ABEC4D521349820505B483E8D48C42EA072`。
- 审核包`manifest.json` SHA-256：
  `B31953DCC31EBB62C697CCF366BC59F5118D1FCB1FF3E0BFA13A63A858810C20`。
- 正式真值SHA-256保持：
  `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`。

## 训练边界

- 只允许进入train；首轮每个batch的合成样本占比不超过30%。
- 用真实val对比有/无合成数据的检测precision、recall和完整场景率；禁止用合成val选阈值。
- 24张来自8个场景，能验证流水线和补充颜色/角度扰动，不能替代跨车辆、跨设备、跨光照的真实数据。
- 当前真实集没有防松线端点和NORMAL/LOOSE状态真值，所以端到端松动状态准确率仍未被证明。
