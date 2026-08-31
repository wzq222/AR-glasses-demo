---
task_id: local
role: Model Error Analyst
objective: Locate the dominant causes of the current accuracy failure.
status: complete
confidence: high
sources_found: 4
acceptance_met: yes
---

## Sources

[L1] `docs/validation/2026-08-28-high-accuracy-validation.md` | LOCAL VERIFIED
[L2] `E:/crrc_vision_data/review-packs/high-accuracy-errors-v2/errors.json` | LOCAL MACHINE REPORT
[L3] `E:/crrc_vision_data/runs/marked-point-proposals-v1/candidate-gate-v1.4.json` | LOCAL MACHINE REPORT
[L4] `E:/crrc_vision_data/runs/marked-point-proposals-v1/union/candidates.json` | LOCAL MACHINE REPORT

## Findings

- 最佳全紧固件YOLOv8s-P2在`precision >= 0.90`时只有`0.3519 recall`；70 FN、4 FP。[L1][L2]
- 70个FN主桶为tiny 33、lookalike 21、dense pipes 8、blur 7、border 4、dark 1。[L1][L2]
- 13/108真值即使阈值降到0.001也没有IoU>=0.5候选；57/108存在候选但分数低于高精度阈值。[L1]
- 更大的YOLOv8m-P2在同一seed下更差，说明当前瓶颈不是单纯容量不足。[L1]
- marked-point开发真值为248框，候选并集覆盖248/248。[L3]
- 并集共有9,567候选；其中8,841为color-only，719为fastener-only，7为两路共同。fastener来源总计726个候选，在当前开发真值上覆盖248/248；颜色来源只覆盖199/248。[L3][L4]

## Analysis

- 当前主要问题是业务目标错配、微小目标缩放和难负例置信度分离，不是换大模型。
- 独立颜色候选带来约92%的候选负担，却没有增加当前开发集召回；它更适合作为候选ROI内的辅助特征。
- 单阶段检测器同时承担高召回和高精度不合算，应将低阈值proposal和ROI复核分开。

## Gaps

- 59张图均来自同一次采集，248/248不能证明跨车辆泛化。
- fastener候选来自离线多源链，不能直接视为手机端单模型能力。

