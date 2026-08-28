# PicoDet Phase B 首轮验证

## 结论

Phase A 银标门已通过，PicoDet-S 416 与 PicoDet-M 416 已在相同 64 个 train 场景、16 个 val 场景和
697 个接受框上完成首轮 80 epoch 训练。两份 best checkpoint 均可独立复评，并通过 Paddle 2.6.2
兼容环境导出为静态推理模型。当前结果只用于 AI 银标内部路线比较，不代表生产准确率。

首轮推荐保留 PicoDet-S 作为手机基线、PicoDet-M 作为精度挑战者。M 的 AP 略高，但模型约为 S 的
2.9 倍；两者 small AP 都为 0，证明下一轮优先级应是离线重叠切片训练和小目标补强，而不是继续扩大
骨干模型。

## 数据门与完整性

- 累计 reviewed COCO：80 个完整独立场景，train 64、val 16，697 框，15 个隔离 uncertain 场景。
- reviewed COCO SHA-256：`8CC7332D16060572D394B4437EB65C367BA6F6D0BCAC205A91EF4761E5F820DA`。
- AI 银标 SHA-256：`736E06A8CF043AB4184E82DE5B67848735995EEB59DC3918B040D53ECD2D0318`。
- 正式真值 SHA-256 在复核、合并、训练和导出后均保持
  `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`。
- 合成训练图 0，验证集全部为真实现场图，场景泄漏 0。

## 固定环境

- PaddleDetection：tag `v2.9.0`，commit `b25522a0f4bde8c80603f3ba5e3472059972e3b5`。
- 训练：Python 3.11、PaddlePaddle GPU 3.2.2、CUDA 11.8 runtime、RTX 3060 6GB。
- 导出/桌面冒烟：Python 3.8、PaddlePaddle CPU 2.6.2、oneDNN、8 threads。
- 两模型统一使用 416 输入、batch 8、80 epoch、固定 seed、相同增强和同一 COCO split。

PaddleDetection 2.9 在 Windows 默认 GBK 读取 YAML，且数据目录包含中文。训练入口使用解析到同一
Git 外资产目录的 ASCII junction `E:\crrc_vision_data` 写入运行时路径；原始数据位置和字节未改变。

## Best checkpoint 复评

| 模型 | AP | AP50 | AP75 | AP-small | AP-medium | AP-large | AR100 | 权重 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PicoDet-S 416 | 0.064 | 0.189 | 0.031 | 0.000 | 0.091 | 0.057 | 0.190 | 4,801,553 B |
| PicoDet-M 416 | 0.071 | 0.197 | 0.045 | 0.000 | 0.077 | 0.071 | 0.192 | 13,967,358 B |

M 在 AP、AP50、AP75 和 AR100 上小幅领先；S 在 medium AP 上更高，模型小 9.2MB。16 个 val 场景
样本量仍小，差值不足以宣称生产优劣。

## 导出与推理冒烟

PaddlePaddle 3.2.2 可训练但无法用 PaddleDetection 2.9 导出 PicoDet，错误是 PIR 与老静态图混用的
`Value/Variable` 类型冲突；关闭 PIR 仍可复现。改用官方兼容范围内 Paddle 2.6.2 后，S/M 均成功
生成 `model.pdmodel`、`model.pdiparams` 和 `infer_cfg.yml`，训练权重无需转换或重训。

代表图 `IMG_20240529_111830.jpg` 的阈值 0.25 冒烟结果：

- S 整图 12 框、12% 重叠切片 32 框；M 整图 15 框、切片 35 框。
- 四组结果经原图坐标检查均无越界框。
- 单次 Python 端到端：S 整图约 212ms、切片约 234ms；M 整图约 242ms、切片约 288ms。
- 静态模型纯 CPU predictor，50 次预热 + 100 次计时：S P50/P95 27.4/30.4ms，M 39.8/48.5ms。

上述延迟来自桌面 CPU，不替代目标 Android 手机的连续 50 次热机 P50/P95。官方推理器
`--save_results` 分支存在未定义局部变量，CPU benchmark 分支错误调用设备同步；验证使用可视化日志
和独立 Paddle predictor 绕开这两个辅助分支，未修改第三方 checkout。

## 下一步

1. 由现有 64/16 场景确定性生成全图 + 2×2、12% 重叠切片训练 COCO，保持按场景 split 和 val 真实性。
2. 用同一 S/M 或先仅 S 重训，目标是 small AP 和 AR100 显著非零；若无提升则扩充真实小目标完整场景。
3. 把 S 静态模型接入 Android runtime 接口，在指定手机连续热机 50 次测端到端 P50/P95。
4. 未获得受控正常/松动状态真值前，ROI 几何层继续输出 `UNCERTAIN`，不报告松动准确率。
