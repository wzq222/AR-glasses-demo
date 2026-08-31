# 防松线松动状态：评估方法、临时阈值与手机模型建议

## 直接结论

现在不能给出“松动识别准确率”，因为真实数据里没有受控松动样本；但可以立刻建立一个有证据的
**临时分诊阈值**，让手机先做到高召回地找出需要人确认的检查点。阈值应作用在透视校正后的相对
转角及其不确定度上，不应作用在YOLO分类分数或原图中两条线的表面角度上。

建议暂定：`3°`为人工复核触发线，`15°`为高疑似位移线；在真实受控样本标定完成前，二者都不直接
输出机械意义上的“已松动”。现有代码测试夹具中的`8°`不是标定结果，而且在24张构造样本上会因
严格`>`比较漏掉6/8个名义8°样本，必须改成双阈值与置信区间协议。

## 为什么这样定

公开方法中，较完整的路线是先检测紧固件，再分割固定侧、活动侧和防松标记，拟合接缝椭圆与两条
标记线，最后根据两条线在物理接缝上的交点计算相对转角。2024年的一项研究在167张分割图上训练
Fast-SCNN，报告平均角度误差1.145°，说明“分割+几何”比直接状态分类更适合作为本项目基线。[1]
另一项Keypoint R-CNN研究也采用关键点、标记椭圆和几何成像计算任意视角下的松动角，而不是直接
预测松/不松。[2]

公开阈值并不一致：列车螺栓防松线专利的一种实现使用15°，但同时把“只检测到一条线”直接当正常，
这对遮挡和漏分割场景并不安全。[3] 另一项透视校正后的钢结构螺栓研究使用2.8°阈值并在其专用数据上
取得高准确率，但该数字不能跨设备直接复制。[4] 因此取整后的3°适合做**高召回复核触发线**，15°
适合做**高疑似优先级线**，中间区域全部交给人，而不是强行二分类。

若同一检查点有紧固验收基准图，可以只用健康重复拍摄估计告警线。公开框架采用健康角度变化均值的
三倍标准差作为上控制限。[5] 本项目建议改用更抗异常值的
`T_point=max(3°, median(|Δθ|)+3×1.4826×MAD)`。这能建立“偏离自身基准”的阈值，但仍不能在没有
松动样本时计算召回率或误报率。

透视校正是硬条件。受控研究显示，未校正时视角超过20°后波动增加，45°视角下误差可达6.5°；校正后
其实验中的最大平均误差约1.1°。[6] 厂商对见证标记的定义也是“显示移动、松动或篡动”，并没有说
它能直接证明剩余扭矩或预紧力。[7]

## 临时状态协议

先计算透视/椭圆校正后的相对角`θ`，并通过端点扰动、分割增强或多帧重复得到95%置信区间
`[θ_low, θ_high]`：

| 条件 | 手机显示 | 自动含义 |
|---|---|---|
| 图像质量失败、两侧归属不清、标记破损、未知拓扑 | 无法判断，请靠近/换角度 | `INSUFFICIENT` |
| `θ_high ≤ 3°`且两侧标记完整 | 未见明显错位，请人工确认 | `LIKELY_ALIGNED`，不是正式正常真值 |
| 区间跨3°，或点估计在3°–15° | 可能错位，请人工确认 | `POSSIBLE_DISPLACED` |
| `θ_low ≥ 15°`且连续两帧/第二视角一致 | 高疑似错位，优先复核 | `LIKELY_DISPLACED` |
| 几何与历史基准都异常，且人工确认 | 已确认错位 | 才允许记录`DISPLACED` |

如果接缝近似圆形、参考尺寸使用直径，3°和15°对应的理论弦位移/直径约为0.026和0.131，可作为角度
结果的交叉校验。不能继续用当前“任意两端点最小距离”替代接缝交点位移。

## 三种实现路线

1. **纯颜色/直线规则**：最快，但锈蚀、铜色、宽涂层和透视会导致大量假线。本项目已经在真实图上
   验证过这条路会误拟合，不推荐作为状态主算法。
2. **ROI直接分类松/不松**：模型最简单，但当前只有合成三态，没有真实松动类；它会学习颜色、背景
   和生成痕迹，也无法解释阈值。暂不推荐。
3. **轻量分割/关键点+几何+历史基准**：推荐。全图轻量检测器保证候选召回；ROI模型输出固定侧、
   活动侧、两段标记、接缝和质量；确定性求解器输出角度、置信区间和原因；有基准图时增加配准变化
   作为第二证据。

手机端ROI模型建议用`MobileNetV3-Small + LR-ASPP`或约1M参数的Fast-SCNN，在256–320像素裁剪上
训练一个共享骨干、多输出头的模型，而不是照搬论文中的三个分割网络。Fast-SCNN本身面向嵌入式
分割；MobileNetV3也是针对手机CPU优化，LR-ASPP版本相对MobileNetV2方案在原论文中更快。[8][9]
导出ONNX后做静态INT8校准，并在P20 Pro上分别测CPU、XNNPACK和NNAPI；官方文档明确指出执行后端
效果依赖具体模型与设备，不能只看桌面基准。[10]

## 如何把临时阈值升级为正式阈值

最小受控实验建议按物理检查点分组，而不是按视频帧随机切分：

- 至少10个真实螺栓/螺母检查点；每个点由维护人员设置`0/2/3/5/8/10/15/20°`并恢复紧固态。
- 每个角度采集正视、左右斜视三种视角，至少两种距离和明/暗两种光照；记录真实角度、维护人员确认、
  固定侧/活动侧、端点和拍摄设备。
- 同一物理点的所有照片只能进入同一个train/val/test分区，防止背景泄漏。
- 阈值选择目标首先是`DISPLACED`召回；任何真实松动被判为`LIKELY_ALIGNED`都按关键错误计数。
- 独立测试至少包含80个松动物理状态组并要求零个“判正常”错误，才能让95%置信下界接近生产讨论
  所需水平；`INSUFFICIENT`可以存在，但必须报告覆盖率。

## 局限

- 3°/15°是文献与测量误差支持的临时分诊线，不是中车设备维护标准，也不代表预紧力阈值。
- 当前19个真实参考点没有受控松动态，无法验证该阈值的召回、精度或跨车辆泛化。
- 曲面管接头、双螺母和夹箍需要各自的拓扑求解器；首版应只开放有几何定义的螺栓头/螺母检查点。
- ImageGen数据可以训练分割外观鲁棒性，不能用于最终阈值和准确率验收。

## 参考文献

[1] Vision-Based Real-Time Bolt Loosening Detection by Identifying Anti-Loosening Lines. https://pmc.ncbi.nlm.nih.gov/articles/PMC11511543/

[2] Detection of loosening angle for mark bolted joints with computer vision and geometric imaging. https://www.sciencedirect.com/science/article/pii/S0926580522003909

[3] CN113469966B: Train bolt looseness detection method based on anti-loosening line identification. https://patents.google.com/patent/CN113469966B/en

[4] 钢结构工程高强度螺栓微小松动视觉检测方法研究. https://zgglxb.chd.edu.cn/CN/10.19721/j.cnki.1001-7372.2024.02.011

[5] Bolt-Loosening Monitoring Framework Using an Image-Based Deep Learning and Graphical Model. https://pmc.ncbi.nlm.nih.gov/articles/PMC7349298/

[6] A novel anti-loosening bolt looseness diagnosis of bolt connections using a vision-based technique. https://pmc.ncbi.nlm.nih.gov/articles/PMC11106340/

[7] DYKEM Cross Check Torque Seal official product documentation. https://www.itwprobrands.com/product/cross-check

[8] Fast-SCNN: Fast Semantic Segmentation Network. https://arxiv.org/abs/1902.04502

[9] Searching for MobileNetV3. https://arxiv.org/abs/1905.02244

[10] ONNX Runtime: Deploy on mobile. https://onnxruntime.ai/docs/tutorials/mobile/
