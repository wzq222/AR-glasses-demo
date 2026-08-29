# 真实困难防松标记与松动状态识别设计

## 1. 目标与边界

本阶段建立两条彼此独立、最后融合的能力：

1. 从现场全图中高召回定位带红、黄等防松标记的检查点。
2. 在证据充分时判断防松标记两侧部件是否发生相对位移。

视觉输出描述的是`relative displacement indicated by witness mark`，不能单凭照片证明螺栓预紧力或
剩余扭矩合格。面向用户的状态固定为：

- `ALIGNED`：防松标记跨越受监控部件，证据充分且未发现相对位移。
- `DISPLACED`：标记两侧部件存在可重复的相对旋转或平移证据，提示疑似松动或拆动。
- `DAMAGED_MARK`：标记开裂、剥落、褪色或污染，形态不能解释为刚体相对运动。
- `INSUFFICIENT`：遮挡、模糊、标记只在一侧、拓扑未知或多路证据冲突，不能判定。

`DAMAGED_MARK`与`INSUFFICIENT`不得自动折算成`DISPLACED`或`ALIGNED`。生产界面不输出“绝对没松”，
只输出“未发现相对位移”并保留证据图。

### 1.1 检查点范围

防松标记并非只用于普通螺栓/螺母。工程资料还明确覆盖压缩管接头、螺纹管接头、管卡和其他按规定
扭矩或位置装配的可旋转组件。项目因此保留两类有效正样本：

1. 螺栓、螺母、螺柱、双螺母等结构紧固点；
2. 有明确螺纹/夹紧运动副的管接头、压缩接头或管卡。

类别名称不决定能否判定。只有标记跨越可识别的moving/fixed两侧，且连接拓扑已知，才允许进入
`ALIGNED / DISPLACED / DAMAGED_MARK`。单边涂点、普通管身色线、热缩管、警示漆、锈迹或无法确认
运动副的红黄色区域只进入`INSUFFICIENT / LOOKALIKE`。客户工艺文件若限定了更窄清单，以客户SOP为
最终范围。

## 2. 选择的技术路线

采用“高召回检测 + 局部多任务模型 + 拓扑几何 + 可选历史基准 + 拒判”的混合路线。

不采用单一端到端`NORMAL/LOOSE`分类器。它容易学习颜色、背景和ImageGen纹理，不能解释跨曲面标记，
也无法区分油漆自然破损与零件运动。

不采用单一二维线段夹角。现场标记可能跨越螺母斜面、螺柱圆柱面、管接头、双螺母或底板；相同的
三维标记在不同视角下不会保持二维共线。

## 3. 系统数据流

```text
现场全图
  -> 标记检查点高召回候选
  -> 局部高清裁剪与质量门
  -> 多任务模型
       paint / moving-part / fixed-part masks
       两段标记端点和部件边界关键点
       topology / visibility / quality / learned-state
  -> 按topology选择几何求解器
  -> 有point_id时与该点历史基准配准
  -> 学习分数 + 几何残差 + 基准差异融合
  -> ALIGNED / DISPLACED / DAMAGED_MARK / INSUFFICIENT
```

### 3.1 全图候选

- 继续使用stride-4小目标检测器作为高召回proposal，不让它直接决定松动状态。
- 第二门只保留`marked_point`，排除无标记紧固件、管路、热缩管、标签、锈斑和红色结构线。
- 全图切片只用于proposal；状态判断始终回到原图高清裁剪，避免切片缩放破坏细线。

### 3.2 局部多任务模型

输入为检查点周围`320×320`或`384×384`裁剪，共享轻量编码器，输出：

- 四类像素：`background / moving_part / fixed_part / paint`。
- 两段防松线各两个端点，以及部件共享边界、转动中心或边界椭圆关键点。
- `topology`：`bolt_head_plate / nut_stud / nut_plate / double_nut / fitting_pipe /
  clamp_pipe / unknown`。
- `mark_role`：`bridges_moving_fixed / moving_only / fixed_only / ambiguous`。
- 清晰度、遮挡、过曝、裁剪完整性与标记可见性。
- 学习得到的四状态概率，仅作为证据之一。

首个实现固定为Fast-SCNN共享编码器加分割、关键点和分类头，模型导出为ONNX。只有Fast-SCNN在真实
val未达门时，才以MobileNetV3-Large + LR-ASPP作为单一挑战者；桌面训练可使用更大教师蒸馏，但手机
端不运行DINO、SAM或LightGlue大模型。

### 3.3 拓扑几何

- `bolt_head_plate`、`nut_plate`：拟合转动件轮廓、部件边界和两段标记，经过局部平面/椭圆投影后
  计算相对角位移。
- `nut_stud`、`fitting_pipe`、`clamp_pipe`：拟合圆柱或连接环的投影椭圆，在归一化环坐标中比较
  两段标记，而不是直接比较图像斜率。
- `double_nut`：分别确定两螺母表面和共享环，两段标记必须绑定到正确零件。
- `unknown`或拟合残差过大：不得套用通用角度公式，直接进入`INSUFFICIENT`或基准图分支。

几何阈值只由真实受控松动实验标定，ImageGen角度不用于确定生产阈值。

### 3.4 历史基准图

二维码/SOP提供`point_id`时，为每个检查点保存紧固验收后的多视角基准图。当前图与基准图配准时：

- 只在固定部件纹理上估计变换，不使用防松线或转动件区域求配准，避免把真实位移对齐掉。
- 手机端先用ORB/AKAZE + RANSAC/ECC；匹配数、内点率或重投影误差不达门则拒判。
- 为同一检查点保留2--3个合格视角，先检索最近视角，再做局部配准。
- 无基准图时使用单图拓扑几何；两条路线不一致时输出`INSUFFICIENT`。

## 4. 状态决策契约

### 4.1 ALIGNED

只有同时满足以下条件才允许输出：

1. `mark_role=bridges_moving_fixed`。
2. 两段标记、共享边界与受监控部件均可见。
3. 图像质量和几何拟合通过。
4. 相对位移小于真实标定阈值。
5. 学习模型、几何求解和基准比较不存在强冲突。

### 4.2 DISPLACED

至少有两个相互独立的证据支持刚体相对运动，例如：

- 投影归一化后的两段标记端点产生一致角位移。
- 转动件相对固定基准发生同方向变化。
- 标记断口形态与转动方向一致。

只看到漆面裂纹、缺口或色差不构成`DISPLACED`。

### 4.3 DAMAGED_MARK与INSUFFICIENT

- 不规则漆面碎裂、掉漆、污染，但没有稳定刚体变换证据：`DAMAGED_MARK`。
- 单边涂点、标记没有跨部件、严重遮挡/模糊/过曝、未知拓扑、几何和模型冲突：`INSUFFICIENT`。
- App给出可执行补拍提示：靠近、稳定、补光或改变视角；第二张仍失败则人工复核。

## 5. 真实困难样本矩阵

困难样本必须来自真实误差桶和实际连接拓扑，不按普通美学“增加噪声”。

### 5.1 困难正样本

- 标记在全图中仅4--15像素。
- 2°、5°、8°等轻微位移，以及明显位移。
- 斜视、俯视、圆柱面和螺母斜面导致的二维伪错位。
- 油污、锈蚀、灰尘、褪色、局部高光和阴影。
- 轻度运动模糊、失焦、JPEG压缩与边界截断。
- 电缆或管线遮挡一段标记。
- 密集相邻紧固件和多条红/黄标记，要求实例归属正确。

### 5.2 困难负样本与拒判样本

- 红色热缩管、端子、标签、文字、结构漆线、锈迹和反光。
- 标记仅存在于一个零件，不能反映相对位移。
- 油漆自然开裂或剥落，但零件没有发生运动。
- 观察角度造成的伪断裂、伪夹角和伪间隙。
- 两个相邻检查点的标记被错误拼成一条线。
- 重涂后的连续标记、维修后缺少可追溯基准等语义风险，强制拒判。

## 6. ImageGen H1生成协议

### 6.1 批次

先生成H1a 24张并完成物理审计，再扩展H1b 96张，总计120张：

- `ALIGNED`困难正常：24。
- `SUBTLE_DISPLACED`轻微位移：24。
- `OBVIOUS_DISPLACED`明显位移：24。
- `DAMAGED_MARK`标记损坏但无部件位移：24。
- `INSUFFICIENT / LOOKALIKE`：24。

H1a优先覆盖轻微位移、漆面损坏、视角伪错位、遮挡和红色相似物，而不是重复既有三态样本。
其固定配比为：困难`ALIGNED` 4张、`SUBTLE_DISPLACED` 6张、`OBVIOUS_DISPLACED` 4张、
`DAMAGED_MARK` 4张、`INSUFFICIENT/LOOKALIKE` 6张。H1b补齐每类24张。

### 6.2 生成约束

- 每张使用真实train检查点裁剪作为参考图，一次调用生成一个局部样本。
- 保留真实紧固件拓扑、材质、锈蚀、油污、镜头透视和工业背景。
- 位移样本必须把标记绑定到moving/fixed两个表面，只移动转动件一侧标记；固定侧不动。
- `DAMAGED_MARK`只改变漆面连续性，不旋转或平移零件。
- `ALIGNED`困难样本允许模糊、高光和遮挡，但其物理状态不变。
- 生成后可做受控缩放、亮度、白平衡、Poisson噪声、运动/失焦模糊和JPEG变化；变换参数写入manifest。
- 所有同一真实来源点及其变体必须留在train同一group，禁止进入真实val或sealed-test。

### 6.3 双层审核

1. 局部审核：拓扑、moving/fixed归属、标记两段、状态和难度是否物理一致。
2. 全图审核：缩回现场尺度后，目标是否真实、尺寸是否符合小目标分布、是否引入明显ImageGen纹理。

任何物理关系不清、合成痕迹明显或状态不能唯一解释的样本保留`REJECTED/UNCERTAIN`，不进入训练。
审核结论绑定参考图、生成图、裁剪图、全图和标注哈希。

## 7. 训练与防过拟合

- 合成样本只进入train，按物理检查点group划分。
- 合成批次比例目标20%，硬上限25%；上一轮25%提高recall但降低precision，因此新一轮必须同时加入
  `DAMAGED_MARK`、`LOOKALIKE`和真实hard-negative。
- detector与state head分开评价；检测框mAP提升不能替代状态准确率。
- 使用三固定seed报告均值、最差值和波动，不以单seed最佳值选结论。
- 阈值只在真实val选择；合成val不参与生产阈值或状态准确率证明。
- sealed-test在模型、阈值、拓扑范围和排除规则全部冻结后只打开一次。

## 8. 验收门

独立真实测试集至少覆盖100--150个跨设备、光照和拍摄距离的场景，并包含同一检查点的受控
`ALIGNED/DISPLACED`成对数据。验收指标：

- marked-point检测recall不低于0.95，precision不低于0.90。
- `DISPLACED` recall不低于0.95。
- 将真实`DISPLACED`错误输出为`ALIGNED`的关键误判率不高于1%。
- 有效bridge-line拓扑的角度MAE目标不高于3°。
- `DAMAGED_MARK/INSUFFICIENT`必须单独计数，禁止从指标分母静默删除。
- 每个真实场景报告完整检查点召回；只报框级mAP不通过验收。
- 目标手机端到端静态照片P95目标500 ms以内；手机型号冻结后实测，桌面GPU数字不能替代。

在独立真实测试通过前，不接入生产Android默认路径，不宣称已实现可靠松动识别。

## 9. 数据与安全边界

- 现场图像、合成图、标注和权重只存放于Git外`E:/crrc_vision_data`。
- `formal truth`保持只读，SHA-256继续固定为
  `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`。
- MIAD只用于研究难例定义；其数据许可为CC BY-NC-SA，不进入商业训练资产。
- 每批生成、审核、训练和评测均记录输入/输出哈希、代码commit、seed和资产来源。

## 10. 参考依据

- Lei等，防松线两阶段分割、椭圆/线拟合和空间投影：
  <https://www.mdpi.com/1424-8220/24/20/6747>
- Deng等，任意视角下mark bolted joint关键点与几何成像：
  <https://doi.org/10.1016/j.autcon.2022.104517>
- MIAD，视角、复杂背景和表面退化对witness-mark检测的影响：
  <https://arxiv.org/abs/2211.13968>
- DYKEM Cross Check，防松胶作为松动/拆动可视指示器的产品定义：
  <https://www.itwprobrands.com/product/cross-check>
- Henkel LOCTITE SF 7414，明确列出compression fittings、studs、nuts、parts和assemblies：
  <https://www.henkel-adhesives.com/sk/en/product/industrial-inks-and-coatings/loctite_sf_7414.html>
- NASA/TRW工程规范，要求nuts and fittings达到扭矩后涂torque stripe：
  <https://ntrs.nasa.gov/api/citations/19710001585/downloads/19710001585.pdf>
