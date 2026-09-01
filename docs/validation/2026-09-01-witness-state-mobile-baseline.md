# 防松线手机状态基线验证

## 结论

本轮完成了防松线状态的安全分诊合同、历史健康基准估计、Git外 ROI 数据合同、MobileNetV3-Small
多头训练链、ONNX 导出和 Python/Android 同源边界测试。实验模型能够在24张合成 train-only 样本上
学习关键点方向，但未通过实验质量门，也不能声称可以可靠判断真实松动。当前用户只授权显式开启的
Android实验阈值人工复核，并未改变质量门结论。

当前可交付的是“可审计的测量与拒判链”，不是生产状态模型。未标定时正式状态始终为
`INSUFFICIENT`；3°/15°只产生人工复核提示。

## 用户授权的 Android 实验阈值路径

- 打包必须同时设置`CRRC_WITNESS_STATE_MODEL_DIR`与`CRRC_WITNESS_STATE_EXPERIMENTAL=1`；只设模型目录时Gradle拒绝配置。
- 打包输入固定为`witness-roi-mobilenetv3-small.onnx`，SHA-256
  `6D42E0D6C5785866DC65077FCD4D5E6EED576689431CA5C3E6649A280A5880BA`；APK内资产名为`witness-roi.onnx`。
- 默认构建保留“松动状态暂需人工确认”拒判文案且不安装候选点击监听；只有实验构建开关为真且状态估计器初始化就绪时，才显示实验阈值文案并开启点击；初始化失败显示“不可用/已禁用候选点击”。
- 候选点击时复制该叠加框对应的原始检测帧。ROI对齐训练`_crop_box`：最长边扩展`1.5`、最小边`16`，`left/top/right/bottom`各自夹取而不把方形向图内平移，因此边缘ROI可为矩形，再拉伸到`320×320`。状态估计期间和结果对话框关闭前暂停候选推理。
- ONNX输入为`[batch,3,320,320]`，ImageNet归一化；输出必须精确为
  `segmentation_logits [batch,4,320,320]`、`keypoint_heatmaps [batch,4,320,320]`和`quality_logits [batch,4]`。固定ONNX的输出元数据除batch外也使用符号维`-1`，Android静态检查接受“`-1`或预期正维”，但运行时解码后数组仍严格校验为`1×4×320×320 / 1×4`；输入通道和空间维始终必须是具体的`3×320×320`。
  四点通道顺序为`fixed_outer/fixed_joint/moving_joint/moving_outer`，以两线段绝对余弦解码`0–90°`点角度。
- 点角度`<=3°`为“倾向正常（待确认）”，`>3° && <15°`为“可疑，建议换角度复拍”，`>=15°`为“高度疑似松动，必须第二视角确认”。
  区间固定为`[max(0,a-6.3), min(90,a+6.3)]`，只作不确定警示，不改变点估计所属分档。这是对“用区间跨阈值改变分档”建议的显式覆盖：当前用户授权和任务都要求简单点角度阈值，所以保持点估计路由。
- 模型缺失、初始化/运行错误、输出错形、任一非有限输出、退化线段或无效裁剪均显示“无法判断，请调整距离/角度后重拍”，不复用上次状态结果。
- 可测量证据门失败关闭：`segmentation_logits`第2通道为防松线，logit `>=0`的像素至少8个；四个空间softmax热图峰值概率均必须`>=0.005`；每个解码端点在半径2px内必须有正防松线mask支撑；两线段长度均至少4px，两joint间距不得超过较短线段的25%。这些是实验拒判下限，不是准确率阈值。
- `quality_logits`的训练目标对24张样本恒为`[1,0,0,1]`，没有负质量标签，因此Android只检查该输出的形状和有限值，不再据此声称或拒判模糊、遮挡或拓扑质量。只有引入这些类型的负标签并完成独立验证后才能启用质量语义。
- 对当前24张合成train-only ROI做只读兼容性检查：四点空间softmax峰值概率全局最低为`0.007343`；所有解码端点到正mask的距离均为`0px`；线段最短`28px`；joint距离/较短线段长度最大`0.1072`。因此上述拒判下限不会额外拒绝这24张已知训练样本，但仍无真实状态准确率意义。
- 没有新增图像、结果或人工确认记录落盘；当前用户已取消该范围。正式`FastenerState`仍为`INSUFFICIENT`。

## 已实现

- Python 和 Android 共用 `test-vectors/witness-triage-v1.json`，精确覆盖3°、15°、跨阈值区间和第二
  视角确认；`LIKELY_* / POSSIBLE_*`不扩展正式四态。
- 未标定配置仍会返回可用角度、区间和复核提示，不再在测量前提前退出。
- 每检查点健康基准采用
  `max(3°, median(abs(delta_theta)) + 3 * 1.4826 * MAD)`，少于5次健康重复拍摄拒绝建线。
- Git外 ROI 清单包含24张、8个来源场景，三态各8张；全部标记为
  `synthetic_geometry_only=true`、`real_state_truth=false`、`sealed_test_opened=false`。
- ROI 网络为约1.4M参数的 MobileNetV3-Small + LR-ASPP式解码器，输出4类掩膜、4个关键点热图和
  4个质量logit；当前只有防松线mask、四关键点和合成质量有监督，固定侧/活动侧/接缝头仍无真值。
- 稀疏mask和热图使用平衡BCE+Dice、空间分布损失和相对角几何一致性损失；训练增强同步作用于图像、
  mask和热图。

## 数据与治理

| 项目 | 结果 |
|---|---|
| ROI manifest | `runs/witness-roi-v1/dataset/manifest.json`（Git外） |
| 样本 | 24张，8个来源场景，train-only |
| 状态分布 | NORMAL 8 / SLIGHT_LOOSE 8 / OBVIOUS_LOOSE 8 |
| source manifest SHA-256 | `FF666BA52427E2F5FE19963B92F706432307281216058FB78C547FC565562B8C` |
| dataset manifest SHA-256 | `6875F76EC60189CC07319B12C29AD61BF9C55FC55B2DDEF0B817C0BC847D2021` |
| formal truth SHA-256 | `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001` |

## 受控训练实验

所有指标都在同一24张合成train样本上计算，只能作为过拟合冒烟，不能当验证集准确率。

| 实验 | mask IoU | 关键点mean / P95(px) | 角度mean / P95(°) |
|---|---:|---:|---:|
| scratch，旧稀疏损失，8 epoch | 0.000 | 228.60 / 276.90 | 90.00 / 90.00 |
| pretrained，stride-8，平衡损失，40 epoch | 0.178 | 2.93 / 5.00 | 4.42 / 9.69 |
| pretrained，stride-4关键点头，40 epoch | 0.163 | 1.69 / 2.83 | 4.00 / 11.47 |
| pretrained，stride-4 + 角度一致性，40 epoch | 0.136 | 1.89 / 3.61 | 3.37 / 6.27 |

调试证据表明：标签端点回算相对原声明角度的平均误差只有0.35°，单样本100步可以收敛；主要上限是
24张/8来源的数据规模、缺少真实固定侧/活动侧/接缝监督，以及短线段上1–3像素误差造成的角度放大。
继续在同一24张上堆损失会增加过拟合风险，因此停止参数试探。

## 导出门

实验导出门固定为：mask IoU≥0.50、关键点P95≤3px、角度mean≤2°、角度P95≤3°。最终模型四项
均未全部通过，标准导出命令正确拒绝；仅使用显式`--allow-failed-gate`生成接口验证模型。

| 项目 | 结果 |
|---|---|
| checkpoint SHA-256 | `837ACAD90A60BC1A14ECAC31CC70A4CDF499AFAE03DCDBEFA7DDAC30CA0863D3` |
| ONNX SHA-256 | `6D42E0D6C5785866DC65077FCD4D5E6EED576689431CA5C3E6649A280A5880BA` |
| PyTorch/ONNX最大绝对差 | segmentation `5.72e-6` / keypoint `7.99e-6` / quality `1.55e-6` |
| ONNX parity | PASS |
| experimental quality gate | FAIL |
| `android_packaging_allowed` | `false` |
| 真机 ROI 时延 | 未测；质量门失败，不把失败模型装机 |

## 回归验证

- 基础Python环境：`361 passed, 3 skipped`；训练环境补齐依赖后：`373 passed`。
- Android：`56 tests, 0 failures, 0 skipped`。
- Debug APK构建成功，65,125,685 bytes，SHA-256
  `B938BAB33B10FB7B95B6213C19D251437C476020FDDC875B0BD386F0FA129007`；该APK没有状态模型。
- formal truth SHA-256复核保持
  `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`。

## Android 实验接入验证

- TDD RED：首次目标测试在缺少`SquareRoi`、`WitnessStateEstimate`、
  `DetectionHitTester`和`OnnxWitnessStateEstimator`时以28个编译错误失败；新增的帧外候选反例后，
  `SquareRoiTest`以1个预期断言失败证明边界拒绝有效。
- 审查修复TDD RED：正式状态字段测试首先以3个`getState()`缺失编译错误失败；补齐字段后，
  空防松线掩膜、低完整性、高遮挡和平坦热图4个反例均按预期先失败。
- 质量审查TDD RED：符号输出维、质量logit仅接口、宽热图、mask/关键点不一致和不连贯joint几何首轮定向测试以5个预期失败进入RED；新`WitnessRoi`和运行交互策略分别先以9个和9个缺失符号编译错误失败，UI源码合同随后以2个断言失败。
- TDD GREEN：目标测试通过；使用当前512 ncnn检测器、128 XNNPACK复核器和实验状态模型环境执行
  `clean testDebugUnitTest assembleDebug --no-daemon`，随后在最终自审修正后复跑`testDebugUnitTest assembleDebug --no-daemon`，
  质量审查修复后再执行同样的干净全量命令，最终结果为`109 tests, 0 failures, 0 errors, 0 skipped`，Debug APK构建PASS。
- 只设`CRRC_WITNESS_STATE_MODEL_DIR`不设实验开关时，`gradlew help --no-daemon`按预期失败并报告
  `CRRC_WITNESS_STATE_MODEL_DIR requires CRRC_WITNESS_STATE_EXPERIMENTAL=1`。
- Debug APK：94,376,108 bytes，SHA-256
  `9134D5036F2438E44364EE2EA0B37072B12DB5CF2524B753D4BA007D2D3044FA`。
- APK内嵌资产SHA-256：检测器param
  `EE68160881FE607CCE87485E569095A917A1511394BE66F39FE7567EFE4C9BB0`，检测器bin
  `ED1448C049809A4E8E2D1D2AFD254AAE66AA4C1238D70B1CA6D9C2835DE9DCEC`，候选复核器
  `FED197A11134DD4358B70EFF64086C050DDECC9B2C484E72AAEB102E4BA563CD`，实验状态模型
  `6D42E0D6C5785866DC65077FCD4D5E6EED576689431CA5C3E6649A280A5880BA`。
- 本轮未安装到手机；单ROI冷/热P50、P95、内存和10分钟稳定性仍待验证。

## 下一训练门

停止继续用同一24张合成图调参。下一批必须采集真实受控检查点：至少10个物理点，每点覆盖
`0/2/3/5/8/10/15/20°`、正视/左右斜视、两种距离和明暗光照，并记录维护人员确认。优先标注至少
200个高质量ROI的固定侧、活动侧、防松线、接缝和四关键点；同一物理点全部留在同一分区。模型先在
独立物理点验证达到上述角度门，再允许进入Android候选点击与真机时延测试。
