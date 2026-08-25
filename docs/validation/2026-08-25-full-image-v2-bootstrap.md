# 全图防松线 V2 标注引导链路验证（2026-08-25）

## 结论

V2已建立100个场景组的代表帧清单、物理紧固件COCO真值骨架、强制训练质量门和12张开放词汇建议框
试跑。GroundingDINO试跑判定为`FAIL`，没有扩展到100张，也没有把建议框写入训练真值或启动训练。

## 代表帧与真值骨架

- 源数据：482张、177个场景组。
- 代表帧：100张来自100个不同场景组；train 80、val 20。
- HSV候选密度覆盖：0候选14张、低密度28张、中密度30张、高密度28张。
- `selection-v2.json` SHA-256：
  `D486FACB0A058CA8EDB146FCDF54F0042F83C71FF6393F6A7EB71B6745B33539`。
- 物理紧固件COCO骨架包含100张图、0个框，全部为`unreviewed`；训练门正确返回
  `UNREVIEWED_IMAGE / NO_ACCEPTED_BOX / INSUFFICIENT_REVIEWED_GROUPS / EMPTY_VALIDATION`。
- `instances.json` SHA-256：
  `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`。

## GroundingDINO试跑

- 模型：`IDEA-Research/grounding-dino-tiny`，固定revision
  `a2bb814dd30d776dcf7e30523b00659f4f141c71`。
- 运行时：Transformers `4.40.2`、PyTorch `2.7.1+cu118`、RTX 3060 Laptop GPU。
- 权重加载检查：missing 0、unexpected 0、mismatched 0。
- 分层试跑：12张、33个建议框；逐框复核接受7、拒绝26，精确率21.21%。
- 12/12张仍有明显未覆盖目标，漏框图率100%。
- 主要错误：把波纹管、滤筒、成束管线或接近整张图的区域当成`pipe_joint`，并对同一紧固件产生重复框。
- 门槛要求精确率至少50%、漏框图率不高于30%，因此结果为`FAIL`。
- `pilot-proposals.json` SHA-256：
  `4AC9A8385D1E0EE2459A20E172C5E36D943143191E8799E8B9375E242A55D121`。
- `pilot-decisions.json` SHA-256：
  `D279FBF0316BE4E35C8DC06EA8D3733368CA37C38D0D28BDBB4EF7DD5E8955D2`。
- `pilot-audit.json` SHA-256：
  `8CFB6DD901C34917791E41B8FA0C42761BF1E3749790EA866CD13521BA58F2D2`。

试跑初次使用了错误的2023权重revision，Transformers报告108个missing和38个unexpected权重，推理被
主动中止。切换到上述固定revision后重新验证为0/0/0；代码现在同时固定Transformers版本并强制检查
完整权重加载，防止静默生成垃圾建议框。

## 用户提供的参考检测权重

用户随后提供了Git外部参考包`Bolt_Detection.zip`。包内检测权重经受限白名单和
`torch.load(weights_only=True)`加载，确认为Ultralytics 8.2.40训练的YOLOv8s Detect：3类、
11,136,761个参数、640输入。权重内可恢复300轮、batch 8等训练参数，但`data/luosi.yaml`、图片、
标签和训练脚本均缺失；配套推理代码依赖的第二阶段分割权重也不在包内。

该检测权重在上述同域12图的640输入下产生92个候选，12/12张有结果，候选裁剪视觉抽查大多落在
真实紧固件或管路连接部位，明显优于GroundingDINO；但在两张不同机柜实拍上，640输入仅产生2个和
0个框，1280输入也仅产生6个和1个框，并漏掉大量明显螺钉。代码硬编码的
`IMG_20240529_111621.jpg`又存在于当前482图中，因此同域结果可能包含原训练场景，不能作为独立精度。

结论：该权重只作为私有离线标注教师和对照基线，不进入仓库、不直接作为手机端生产模型。正式使用
还受原始数据来源、权重授权和Ultralytics AGPL-3.0/商业许可约束。完整静态分析、盲测JSON和叠加图
保存在`CRRC_VISION_DATA_ROOT`下，未加入Git。

## 代码验证

```powershell
$env:CRRC_VISION_DATA_ROOT='E:\Work\京新数智\识动hicool\中车眼镜数据资产'
.\.venv\Scripts\python.exe -m pytest ml\tests -v
```

结果：37项Python测试全部通过。代表帧选择可重复、场景组不交叉；训练真值仍为未就绪，没有生成模型
checkpoint，也没有报告检测准确率或手机端延迟。

## 下一步

进入人工框标注，不再尝试把开放词汇建议框批量扩展。优先用用户提供的YOLOv8s参考权重为100张代表
帧生成同域候选，再结合已确认的151个HSV色标定位锚点逐图补漏、删错并核实三类含义；只有至少80个
场景组完成整图复核后，才编写和执行PicoDet-S/M训练计划。
