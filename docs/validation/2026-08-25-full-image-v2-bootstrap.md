# 全图防松线 V2 标注引导链路验证（2026-08-25）

## 结论

V2已建立100个场景组的代表帧清单、物理紧固件COCO真值骨架和强制训练质量门。GroundingDINO
12张试跑判定为`FAIL`；随后把用户提供的YOLOv8s参考权重隔离为Git外标注教师，完成100张候选生成
与全量视觉复核。教师候选明显优于开放词汇模型，但整图仍普遍漏检，100/100张都没有达到完整真值
标准；正式真值未改变，也没有启动训练。

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

## YOLOv8s参考教师100图运行与复核（2026-08-26）

- 隔离运行时固定Ultralytics `8.2.40`、PyTorch `2.7.1+cu118`，检查点仅经静态全局枚举、前缀
  白名单和`torch.load(weights_only=True)`加载；权重SHA-256为
  `65DFC280A31C6FA177EF06086BF2CA195B927DE9AF7E8739D3AF7965ECCD3315`。
- 输入为`selection-v2.json`的100张代表帧，参数为640输入、confidence 0.25、IoU 0.70、CUDA 0；
  共生成731个候选，教师class 0/1/2分别为186/164/381个，94张有框、6张零检测。
- 31页候选裁剪和13页整图叠加已逐页查看。候选层记录为691个`accept`、39个`needs_manual`、
  1个`reject`；这里的`accept`只表示视觉上像紧固件或辅助连接件候选，不代表可直接训练。
- 整图层100/100均为`needs_manual`、0张证明完整；6张零检测图仍能看到巡检目标。整图复核还发现
  大量可见未框目标，因此不能从731个候选计算召回率，也不能把候选级接受率解释为模型准确率。
- 桌面RTX 3060 Laptop GPU上，首张含初始化的wall time为3,227.258 ms；其后99张wall time均值
  23.751 ms、P95 29.283 ms，纯推理均值8.764 ms、P95 11.919 ms。这只是桌面CUDA标注吞吐，
  不是目标手机端到端性能。
- `raw-predictions.json` SHA-256：
  `7173D1C064C12D38A4B3E6D99CC044F830A46D312D051E2C3C71F55ACE6C878E`。
- `instances.proposals.json` SHA-256：
  `DB7F9564EB7FDA71BE1706AA9B3CF8D197D6FBD4E6AC7A9E3872B7E21A331606`。
- `ai-review-v1.json` SHA-256：
  `4FCB8010BF2AA40AFE5042A24B745BBAC35065E604C51EE2183458C0EFEFE63F`。
- 正式`instances.json`运行前后SHA-256均为
  `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`；训练门保持关闭。

## 代码验证

```powershell
$env:CRRC_VISION_DATA_ROOT='E:\Work\京新数智\识动hicool\中车眼镜数据资产'
.\.venv\Scripts\python.exe -m pytest ml\tests -v
```

结果：72项Python测试全部通过。代表帧选择可重复、场景组不交叉；参考权重安全加载、预测契约、
多尺度/切片元数据、独立来源融合、同场景传播硬门、Codex两遍审核和银标拒绝门均有自动测试。

安全自动标注编排另做了拒绝性实测：把仅覆盖100张的旧teacher结果提交给482张全量manifest，程序
明确返回`missing=382 extra=0`，未创建输出目录；正式`instances.json` SHA-256仍为
`B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`。这证明链路已具备评估准备度，
不代表全量银标已经生成，也不代表检测精度或手机端性能达标。

随后完成482图全量运行：三种尺度、整图加四个12%重叠切片共生成15,463个teacher框；与1,993个
HSV候选和1个严格通过的时序传播框合并为5,535个融合候选。305条同场景相邻边中34条通过
ORB/RANSAC质量门，因传播还要求来源图已有teacher+HSV高共识，最终仅加入1个时序候选。复核包包含
482张整图、5,535张2倍上下文候选图和61个首轮任务批次；22张零候选图仍列入整图审核。

- 全量teacher输出SHA-256：`E7B3D068C14926BD1CB68451F9221E3D816EABB9A983741AF8CD4BD85D6A923C`。
- 融合候选SHA-256：`CD34CDFC2712B152F6E49FA4B529807F49E651644C82B9980FD7A62F20430B82`。
- 复核包manifest SHA-256：`DCB84521BCA96B4F69DE0CC6636663EF84A695E6F50CB589968E3FA6159BD912`。
- 正式真值SHA-256仍为：`B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`。

## 下一步

由Codex完成61批首轮整图/逐候选复核，并对新增或调整框执行隐藏首轮结论的第二遍。只有银标质量门达到至少64个
train和16个val完整场景组，才执行PicoDet-S/M训练；当前100张旧结果仍不得作为完整训练真值。
