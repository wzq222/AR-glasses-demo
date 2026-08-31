# 防松线手机状态基线验证

## 结论

本轮完成了防松线状态的安全分诊合同、历史健康基准估计、Git外 ROI 数据合同、MobileNetV3-Small
多头训练链、ONNX 导出和 Python/Android 同源边界测试。实验模型能够在24张合成 train-only 样本上
学习关键点方向，但未通过实验质量门，禁止打包进 Android，也不能声称可以可靠判断真实松动。

当前可交付的是“可审计的测量与拒判链”，不是生产状态模型。未标定时正式状态始终为
`INSUFFICIENT`；3°/15°只产生人工复核提示。

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

- 基础Python环境：`360 passed, 3 skipped`；训练环境补齐依赖后：`372 passed`。
- Android：`56 tests, 0 failures, 0 skipped`。
- Debug APK构建成功，65,125,685 bytes，SHA-256
  `B938BAB33B10FB7B95B6213C19D251437C476020FDDC875B0BD386F0FA129007`；该APK没有状态模型。
- formal truth SHA-256复核保持
  `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`。

## 下一训练门

停止继续用同一24张合成图调参。下一批必须采集真实受控检查点：至少10个物理点，每点覆盖
`0/2/3/5/8/10/15/20°`、正视/左右斜视、两种距离和明暗光照，并记录维护人员确认。优先标注至少
200个高质量ROI的固定侧、活动侧、防松线、接缝和四关键点；同一物理点全部留在同一分区。模型先在
独立物理点验证达到上述角度门，再允许进入Android候选点击与真机时延测试。
