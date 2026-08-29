# 防松线合成数据训练消融

## 结论

24张经全图审核的ImageGen防松线样本以每batch `3 real + 1 synthetic`进入YOLOv8s-P2训练后，
在同一真实val上提高了召回和mAP，但降低了precision。该结果证明合成样本对缓解漏检有正向信号，
尚未达到“高precision + 高recall”的生产门，不采用该权重上线，也不据此声明能判断真实防松线松动状态。

## 公平性约束

- control和mixed均从同一个预训练权重开始，seed为`20260829`，输入为640，batch为4，最多20 epoch。
- 两组每epoch均为39个batch；mixed每个batch固定3张真实视图和1张合成视图，合成占比25%。
- mixed原始训练COCO包含78张真实全图和24张合成全图；合成图只进入train。
- 两组真实val物化后均为19张图、108个实例，38个图像/标签文件逐文件SHA-256一致；合并摘要为
  `046DFC4CF1EC6638991435FB4813C6311EA565A9C221276A73BF3447839729AE`。
- sealed-test未打开，formal truth未修改。

## 同口径真实val结果

| 训练臂 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| real control | 0.7093 | 0.5463 | 0.5900 | 0.2936 |
| real + 25% synthetic | 0.6175 | 0.6278 | 0.6259 | 0.3095 |
| mixed - control | -0.0918 | +0.0815 | +0.0359 | +0.0158 |

mixed在第4 epoch取得best，因patience 15在第19 epoch正常早停。Ultralytics 8.2.40随后在PyTorch
2.7.1的optimizer剥离阶段触发`Weights only load failed`。原始checkpoint未覆盖；只对本机自产、哈希
固定的best复制件执行受控optimizer剥离，再由受限安全加载器完成评测。

## 完整性证据

- control best：`DB13A92A8D66BD8229E62F55534964929B3B5EAC91106B1F148F6FA51A31D3F7`
- mixed raw best：`17ECD0AB23D50995F7E5E6D9AE4034DAF0A4E9FFF601BDBA0B4BEF0BEF8EFB14`
- mixed inference best：`73530EBB98647C638BDBE0B8CA199078B2ECB0B313F5BFFB751A6062AF11C929`
- mixed COCO：`743C58C87C4FC4E2842AAFE5CC7A6525CC16D265759A454DB63F935BB0EC7E80`
- 真实val COCO：`507E724EBFA22AE4DD9DEF33CDA14CC3DE9917EE537C67DE757B1BEE5370415D`
- formal truth：`B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`
- Git外机器可读报告：`E:/crrc_vision_data/runs/synthetic-ablation-evaluation-v1/report.json`

## 能力边界与下一步

这是单seed、小规模、同源真实val的检测框消融，不是独立泛化结论。现有真实数据仍没有防松线端点和
NORMAL/LOOSE状态真值，因此本轮不能评价松动判断准确率。下一轮应保留25%合成上限，增加真实困难
负样本和至少100--150个跨设备/光照场景，再做三seed复验；同时采集同一紧固件NORMAL/LOOSE受控
成对照片并标注两段防松线端点。只有独立真实测试同时通过precision、recall和完整场景门，才进入手机
runtime和现场验证。
