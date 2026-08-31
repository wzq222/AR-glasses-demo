# 带防松标记检查点模型恢复验证

日期：2026-08-31

## 结论

同源开发验证门首次通过：在17个场景、75个`marked_point`真值上，E1单类YOLO-P2候选、
E4三分类MobileNetV3-Small复核、双分数保底和IoU 0.3去重的组合保持75/75覆盖。后续三固定seed
复核发现单模型负担不稳定；三个seed的等权几何均值为17.59个/图，单模型权重平均挑战者为
19.71个/图，两者均保持75/75。当前推荐保留单模型权重平均版本进入跨设备挑战，不直接接入生产。

这不是生产准确率。阈值在同一17场景开发验证集上选择，数据来自同一次采集；sealed test没有打开，
没有跨车辆、跨设备或手机端时延证据。最密集单图仍有51个候选，禁止硬截断为Top-20，因为该策略
只能覆盖69/75真值。

## 实验结果

### E1：单类YOLO-P2候选

- 训练：30个train场景、173框；物化为60个训练视图、238框；17个val场景、75框。
- 最优epoch 23；默认验证P 0.587、R 0.560、mAP50 0.537、mAP50-95 0.258。
- 全图切片融合在proposal覆盖语义下为75/75，阈值0.0025412512477487326。
- 候选负担为62.94个/图，召回门通过、负担门失败。
- 权重SHA-256：`44D00F6BD7AC64DBDFBFBEF1A613F643C19FBE09EC56E12F93419CB9766870FF`。

严格IoU 0.5召回仅0.88，原因是若干真值框为宽松180×180框，而模型输出为紧框。业务候选门因此使用
“候选中心落入真值，或IoU≥0.10，或较小框包含率≥0.50”；严格IoU指标仍保留为诊断，未删除。

### E2：困难负样本

- 50%困难负样本版本在conf 0.001只有0.9733覆盖；降至0.0001恢复100%时恶化到88.18个/图。
- 10个困难负样本微调版本恢复100%覆盖，但为89.47个/图。
- 已修正微调重复warmup问题，并保持`weights_only=True`与精确numpy安全全局允许列表。
- 两个E2均失败，E1继续作为候选模型。

### E3/E4：ROI复核器

二分类E3使用635个训练ROI和全部1070个E1验证候选，100%真值覆盖时为49.18个/图；与E1双阈值
融合后为35.59个/图。3×上下文版本恶化为53.06个/图，说明扩大背景会稀释细小漆线特征。

复核发现二分类把所有未撞上正框的候选合并为同一负类，丢失原双审的
`unmarked_fastener/lookalike`语义。E4改用哈希固定的`review-complete-v1.4.json`构造三分类集：

- train：173 marked、198 unmarked、300 lookalike；
- val：75 marked、99 unmarked、2035 lookalike；
- 30/17场景隔离；未匹配权威审查决策的候选不进入训练；
- MobileNetV3-Small权重SHA-256：
  `E6EB45369886507A2BF5817BE01D958D8FB1642A75CC5BE5CC7EA4E8D613787D`。

E4原计划12轮；在第2轮保存当前最佳权重后，因一次已中止实验的遗留进程争抢GPU而主动结束本次
探索训练并立即进行端到端门验证。该权重是完整可加载checkpoint，但本轮不是多seed收敛结论；后续
扩充真实数据后必须重新做固定种子与独立测试。训练脚本现已增加逐epoch进度落盘，防止再次丢失曲线。

E4回灌全部E1验证候选后的逐级结果：

| 阶段 | 真值覆盖 | 候选/图 |
|---|---:|---:|
| E1融合候选 | 75/75 | 62.94 |
| E4 marked分数门 | 75/75 | 29.82 |
| E4或E1双通道保底 | 75/75 | 25.88 |
| 双通道 + IoU 0.3去重 | 75/75 | 18.35 |

最终312个候选中，106个候选与现有正真值相关，206个按现有正真值不相关。该candidate relevance不是
业务precision：多个候选可覆盖同一真值，且验证真值只定义检查点框，不定义防松线端点与状态。

### E4三seed稳定性与单模型权重平均

按固定seed `20260828/20260829/20260830`、相同数据、6轮和相同门完成复训；最佳轮分别为3、3、2。
回灌同一1070个E1候选后：

| 模型 | 真值覆盖 | 候选/图 | ≤20门 |
|---|---:|---:|---:|
| seed 20260828 | 75/75 | 22.47 | 失败 |
| seed 20260829 | 75/75 | 17.29 | 通过 |
| seed 20260830 | 75/75 | 23.24 | 失败 |
| 三seed等权算术均值 | 75/75 | 18.71 | 通过 |
| 三seed等权几何均值 | 75/75 | 17.59 | 通过 |
| 三seed权重平均单模型 | 75/75 | 19.71 | 通过 |

因此高召回在三个seed均稳定，但单seed候选负担不稳定，`single_model_all_passed=false`。等权几何均值
不拟合额外权重，能压制只有一个seed高分的干扰候选，但手机需三次ROI推理。权重平均只运行一个
MobileNetV3-Small，浮点状态等权平均，BatchNorm计数取最大，其他非浮点状态不一致即拒绝；其权重
SHA-256为`40E3BCF8114C0C3754E329B54775D02471C22307F9656FF47BEC56D9BA622AE5`。该版本距离20门
仅余0.29个/图，不构成跨域稳健证据；仅作为下一批真实数据的单模型挑战者。

## 可复现资产

- E1门：`E:/crrc_vision_data/runs/marked-point-p2-e1-pilot/gate-fused.json`
- E4语义数据：`E:/crrc_vision_data/runs/marked-point-verifier-e4/semantic-dataset/manifest.json`
- E4权重：`E:/crrc_vision_data/runs/marked-point-verifier-e4/mobilenetv3-small-semantic/best.pt`
- 端到端报告：`E:/crrc_vision_data/runs/marked-point-verifier-e4/e1-candidate-predictions.json`
- 三seed与等权集成报告：`E:/crrc_vision_data/runs/marked-point-verifier-e4/multiseed/ensemble-report.json`
- 单模型权重平均挑战者：`E:/crrc_vision_data/runs/marked-point-verifier-e4/multiseed/model-soup-final/best.pt`
- formal truth SHA-256：
  `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`，验证前后未变。
- `python -m pytest ml/tests -q`：302 passed。

## 能力边界与下一门

当前实现解决的是“全图尽量不漏地提出带防松标记检查点，并把人工复核量压低”。它尚未实现可靠的
`NORMAL/LOOSE/UNCERTAIN`状态判断。松动状态必须新增同一物理实例的受控正常/位移成对图，标注固定侧、
活动侧两段漆线端点、可见性、遮挡与经维护人员确认的状态；单张历史图或ImageGen不得充当状态真值。

下一门是新增至少100–150个跨车辆/设备/光照真实场景，冻结模型与阈值后建立独立sealed test；通过后
再导出移动端模型并在目标手机上测全链路P50/P95、热机和内存。当前结果不允许直接宣称高生产精度。
