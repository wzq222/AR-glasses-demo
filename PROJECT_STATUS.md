# Project Status

## Purpose

“中车眼镜”是面向轨道交通设备巡检的 AR 眼镜辅助作业项目，近期目标是在固定硬件和受控场景下
跑通三步巡检 SOP：二维码打卡、防松线状态判断、万用表读数，并形成带时间与证据照片的结果记录。

## Verified Current State

- 2026-08-25：仓库从 `https://github.com/wzq222/AR-glasses-demo.git` 克隆，保留上游 Git 历史；
  基线提交为 `510e5f5`。
- 代码包含单个 Android 手机端模块，使用 Java、BLE、WiFi Direct、HTTP 和 K900 SDK。
- `GlassBleService` 实现眼镜发现/连接、进入照片导入模式、WiFi Direct 组网、HTTP 下载和退出导入模式。
- `ImageAnalyzer` 定义二维码、防松线、万用表三个接口，但 `DefaultImageAnalyzer` 均为占位实现；
  当前 UI 没有调用 `Vision`。
- 2026-08-25：482张现场全图已在Git外完成SHA-256/尺寸/清晰度清单，按时间与pHash划为177个场景组；
  训练424张、内部验证58张，场景泄漏0。
- `ml/` 已实现可复现的清单、分组、HSV色标候选、COCO输出、分层复核包和训练质量门。
- HSV候选v2生成1,993个候选；AI逐候选复核60张/257个候选，接受151、拒绝97、需人工9，精确率
  60.89%，低于80%训练门；D-FINE-N训练未启动。
- Android已加入全图多目标结构和防松线几何公式；关键点低置信度或现场阈值未标定时强制返回
  `INSUFFICIENT`。正式状态契约为`ALIGNED / DISPLACED / DAMAGED_MARK / INSUFFICIENT`，旧
  `isNutLoose`仅保留兼容；这部分是模型无关核心，尚未接入真实检测推理。
- 2026-08-25：V2从177个场景组确定性选择100个代表帧（train 80/val 20），生成物理紧固件真值
  骨架；当前100张全部未复核、0个真值框，训练门继续拒绝。
- GroundingDINO 12张试跑产生33个建议框，逐框复核仅接受7、拒绝26，精确率21.21%，且12/12张
  存在明显漏框；试跑门为FAIL，未扩到100张，也未污染真值。
- 用户提供的Git外部参考权重确认为YOLOv8s三类检测模型；同域12图在640输入产生92框且候选质量较高，
  但两张不同机柜实拍仅产生2/0框，跨场景漏检严重。该权重只作私有标注教师；原数据、类别YAML、
  第二阶段分割权重和来源授权缺失，且存在Ultralytics AGPL-3.0/商业许可约束。
- 2026-08-26：Git外YOLOv8s参考教师已在100张V2代表帧上受限加载并完成推理，产生731个候选；
  AI逐页复核31页候选裁剪和13页整图叠加，候选层691接受、39需人工、1拒绝，但整图层100/100均
  需人工补全、6张零检测。正式真值SHA-256前后不变，训练门保持关闭。
- 2026-08-26：安全自动标注Phase A代码链路已实现：teacher多尺度整图/2×2重叠切片、HSV独立锚点、
  同场景ORB/RANSAC传播硬门、多来源融合、Codex首轮整图复核与盲审第二遍、整图银标质量门和拒绝式
  导出。73项Python测试通过；用旧100图teacher输出对482图清单做拒绝性实测时正确报告缺382图、
  不创建半成品目录，正式真值SHA-256保持不变。
- 2026-08-26：482图的640/960/1280整图+2×2切片teacher全量推理已完成，产生15,463个原始框；
  与1,993个HSV候选及1个通过硬门的时序候选合并后得到5,535个融合候选。ORB/RANSAC检查305条
  同场景相邻边，仅34条通过几何门，最终只传播1个有高共识来源的框。Git外Codex复核包已生成
  482张整图、5,535张候选上下文和61个首轮批次；22张图仍为零候选，必须做整图审核。
- 2026-08-26：Codex首轮视觉审核已完成4/61批，覆盖前32张整图和636/5,535个候选；累计判定
  299个接受、174个拒绝、163个不确定。四份Git外审核JSON均通过review契约、候选ID全覆盖、
  资产SHA-256一致性检查；不确定项未进入银标，正式真值SHA-256保持不变。
- 2026-08-26：已定位前四批`32/32`整图不确定为审核流程缺陷，而不是32张均不可判：其中5张候选层
  已无未决项，但旧1000像素整图不能证明无漏检；同时首轮补框/修框被校验器立即要求已有盲审二遍，
  形成无法保存待二审几何的死循环。现已加入`pending_second_pass`、4块重叠高清漏检扫描、明确补框契约，
  并按100个代表场景生成Git外`safe-auto-v2`包：100整图、400扫描块、1,711候选、13批，真值哈希未变。
- 2026-08-26：安全自动标注V2已用4个独立场景跑通校准闭环：2张首轮直接完整、1张真实运动模糊
  保留不确定、1张发现4个漏框并经隐藏结论二审完成；最终得到3张完整图/7个接受框/1张不确定图。
  校准COCO哈希为`6E9A4B4B2F7985712D91206962B3A8F2F0041C6EA5312D9900954392DC81C69F`；
  银标门只报告train/val场景数量不足，没有未决框、非法框或场景泄漏错误。新版`safe-auto-v2.1`
  100图包已加入边界截断、微小背景和重复子框排除策略，仍为100整图/400扫描块/1,711候选/13批。
- 2026-08-26：`safe-auto-v2.1`首个扩量批已完成全图扫描、候选决策与隐藏结论二审：新增ID 25、434两个
  train独立场景和5个接受框。校准与扩量批通过可重复图像ID检查后合并为5个完整场景/12框/1个真实不确定场景；
  累计COCO哈希为`138BEDE3E58111690AF311B02049758B7987B7339D757E599543B1845E32ADD3`。银标门仅拒绝
  train 4/64、val 1/16的数量不足，正式真值SHA-256保持不变。`merge_review_documents`现拒绝跨批重复图像，
  并保留二审输入契约，避免手工拼接审核JSON。
- 2026-08-28：累计 reviewed COCO 已达到8个完整场景（train 5/64、val 3/16）和25个框；银标门
  仅剩训练/验证场景数不足，当前缺口为train 59、val 13。该累计结果是审核进度的权威口径，正式真值
  仍未修改。
- 2026-08-28：已定位候选膨胀来自IoU单链聚类与HSV固定`180×180`标记窗口。complete-link首轮v2
  因过严被真实数据门拒绝并保留审计；最终v2.2采用教师代表锚点、HSV唯一归属、歧义保留和簇代表框
  `IoU >= 0.75`末级去重。482图从v1的5,535个融合候选降到4,667个；图0007从15降到12、
  图0047从10降到7、相邻实体反例图0011从37降到25。三张全图模型复审未发现跨实体合并，
  正式真值SHA-256仍为`B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`。
- 2026-08-28：Git外`review-packs/safe-auto-v2.2`已按原100个代表场景重建，包含100张整图、
  400张高清漏检扫描块、1,458个候选和13个首轮批次；旧v1/v2/v2.1资产均未覆盖。Python完整
  回归当前为100项。
- 2026-08-28：`safe-auto-v2.2`固定批次001已完成8个此前未审核场景的高清全图扫描、71个候选首审
  和7张几何盲二审；新增6个完整场景/33框，图37、58因遮挡保留`uncertain`。跨候选版本改为在
  reviewed COCO层安全合并，拒绝重复图像与重复场景并重排annotation ID。累计达到14个完整场景/
  58框（train 10/64、val 4/16），累计uncertain为9；银标门仅剩train/val场景数不足，正式真值
  SHA-256保持`B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`。
- 2026-08-28：`safe-auto-v2.2`固定批次002已审核8个新场景、65个候选并对4张补框图执行隐藏首审
  结论的几何盲审；新增5个完整场景/29框，图75、375、441因暗部、遮挡或锚点归属不清保留
  `uncertain`。累计达到19个完整场景/87框（train 14/64、val 5/16），累计uncertain为12；累计
  reviewed COCO SHA-256为`68A96293D9D4C81B73573C99B20481B76D393FF2AFE903991C0A1C49A8905AFD`。
  银标门仍仅拒绝train/val场景数不足，正式真值SHA-256保持不变。
- 2026-08-28：`safe-auto-v2.2`固定批次003按清晰度、亮度和实际候选数复筛后完成6个train、2个val
  独立场景的高清全图扫描、28个候选首审，并对6张补框图执行隐藏首审结论的几何盲审；新增8个
  完整场景/26框、0个新增uncertain。累计达到27个完整场景/113框（train 20/64、val 7/16），
  累计uncertain仍为12；银标门仅拒绝train/val场景数不足，正式真值SHA-256保持
  `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`，100项Python测试通过。
- 2026-08-28：`safe-auto-v2.2`固定批次004完成6个train、2个val新场景的高清全图扫描和48个候选
  首审；4张补框图共21个提议进入隐藏首审结论的几何盲审，二审接受20、拒绝暗部伪目标1。
  本批新增8个完整场景/40框、0个新增uncertain；累计达到35个完整场景/153框（train 26/64、
  val 9/16），累计uncertain仍为12。累计reviewed COCO SHA-256为
  `8966C5245F360DB280F8461E248E4628429C0D904EEBD9DE4B5BDBCA1DF74874`；银标门仅拒绝train/val
  场景数不足，正式真值SHA-256保持不变，100项Python测试通过。
- 2026-08-28：停止30分钟自动批次后连续完成剩余48个代表场景审核；其中45个达到整图完整门，3个
  因暗部模糊或成组遮挡保留`uncertain`。累计达到80个完整独立场景、697框（train 64、val 16），
  累计uncertain 15；AI银标门正式PASS并导出`silver-gate-cumulative-013`。累计reviewed COCO
  SHA-256为`8CC7332D16060572D394B4437EB65C367BA6F6D0BCAC205A91EF4761E5F820DA`，正式真值哈希未变。
- 2026-08-28：PicoDet-S/M 416在相同64/16银标场景上完成80 epoch首轮训练和best权重复评。S/M的
  AP分别为0.064/0.071，AP50为0.189/0.197，AR100为0.190/0.192，small AP均为0；best权重约
  4.8MB/14.0MB。两模型均通过Paddle 2.6.2兼容导出、整图及12%重叠切片推理冒烟。桌面CPU纯模型
  P50/P95为S 27.4/30.4ms、M 39.8/48.5ms；这不是手机指标，也不是生产准确率。
- 2026-08-28：完成小目标根因修复和P2挑战者对照。PicoDet-S加入保留负样本、2×2重叠
  切片训练、训练周期内assigner切换后，全图+切片原始双类AP50为0.285，物理目标合并
  口径为0.448。stride-4 YOLOv8s-P2在相同16个全图银标场景上使用同一合并口径后，
  640微调版达到AP 0.288、AP50 0.605、AR100 0.479；阈值0.20时precision 0.641、
  recall 0.584。这是当前最优内部基线，仍未达生产准确率。固定640 ONNX已导出到
  Git外`exports/android/yolov8s-p2-v3-640`并通过ONNX结构校验；尚未接入Android或做手机实测。
- 2026-08-28：高准确率V2数据已冻结为train 78场景/638框、val 19场景/108框、sealed-test
  30场景/276框。S-P2三个固定种子在`precision >= 0.90`下的full recall为0.1944/0.2222/
  0.2500，fused recall为0.2037/0.3519/0.2593，所有运行的完整场景率均为0/19，且种子
  稳定性未过门。最佳为seed 20260829 fused：precision 0.9048、recall 0.3519。
- 2026-08-28：单一M-P2挑战者完成30 epoch；同seed的full/fused recall分别为0.1667/0.0833，
  未达到相对S-P2至少+0.03的继续条件，故停止其余M种子和蒸馏。其训练结束时遇到Ultralytics
  8.2.40与PyTorch 2.7.1的optimizer剥离兼容问题，但本地best checkpoint已通过受限安全加载
  完成评估。验证门失败，sealed-test保持未打开，Android集成继续关闭。
- 2026-08-28：最佳S-P2 fused的严格误差包有70 FN、4 FP；主桶为tiny 33、lookalike 21、
  dense pipes 8、blur 7、border truncation 4、dark 1。现有1,022个目标框没有防松线端点或
  正常/松动状态真值，因此只能证明几何判定单元测试通过，不能声称端到端松动识别可用。
- 2026-08-29：方案2的带防松标记检查点闭环已完成代码与开发集审计。修复了“先去重后剔除”
  吞掉全图补框和二审完整性集合判定两个缺陷。Git外 `marked-point-v1.4` 有30个train、
  17个val完整场景和248个标记点，12张因模糊/遮挡/边界归属不清排除。候选门
  248/248，但这仍是同源开发集指标，不是生产精度。train低于64场景且没有防松线端点/
  NORMAL/LOOSE状态真值，训练门保持FAIL。209项Python测试通过，正式真值哈希保持不变。
- 2026-08-29：ImageGen防松线合成试点达到全图门。最终保留8个train来源场景，每个场景生成
  NORMAL/SLIGHT_LOOSE/OBVIOUS_LOOSE三态，共24张局部和24张现场全图，三类各8张。防松线像素
  来自ImageGen供体，仅做分段仿射重定位；全图阶段只迁移涂料像素，不再搬运异种紧固件外观，
  并以保守门清除原红/黄线。mark-only检测框严格保留真实COCO框，训练分区、来源图和背景图均校验
  SHA-256；24张经局部和整图双层复核后全部APPROVED，结论绑定全图、裁剪图和审核包哈希。相同种子
  复跑的内容哈希均为`68C2A420E151329C9CF57D894EEE39243B8FC8CA26119F5953F27DD7BB540016`。
  严格全图审计0错误，248项Python测试通过，正式真值SHA-256仍为
  `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`。
- 2026-08-29：完成真实control与25%合成批次的YOLOv8s-P2单seed公平消融。相同19张/108实例
  真实val上，mixed相对control的recall由0.5463升至0.6278、mAP50由0.5900升至0.6259、
  mAP50-95由0.2936升至0.3095，但precision由0.7093降至0.6175。合成数据对漏检有正向信号，
  但精确率与召回率均未达到生产门；sealed-test未打开，正式真值哈希未变。mixed在第19 epoch
  正常早停，PyTorch 2.7.1最终剥离兼容问题已增加早停回归保护并保留原始权重。
- 2026-08-31：H1真实困难状态任务完成三轮上限审核。24个固定job共41条ImageGen尝试，18个job
  至少有一个局部样本通过，6个job仍未通过；第三轮6张全部REJECTED。根因是生成结果不能可靠形成
  moving/fixed相对位移，以及`ref-12`多检查点裁剪造成拓扑标签歧义。H1a固定配比门保持关闭，失败图
  未进入训练真值。H1审核包提供像素保真的1×/2×/4×证据视图；真实marked-point复核包在有界状态
  复核时可按需生成同样视图，不会为9,567个首轮原始候选默认扩张资产，也不使用生成式超分辨率。
  正式真值SHA-256保持不变。后续将ImageGen限制为外观困难增强，真实松动阈值只用受控
  ALIGNED/DISPLACED成对采集标定。
- 2026-08-31：完成YOLO准确率恢复复盘。现有高准确率误差的主矛盾为tiny与lookalike，扩大到M模型
  无效；marked-point开发集的fastener来源仅726个候选即覆盖248/248真值，而color-only产生8,841
  候选。下一轮固定为两阶段：单类marked-point YOLOv8s-P2做低阈值proposal，原图+切片训练和定向
  hard-negative mining；MobileNetV3在原图ROI内复核有效标记/无标记紧固件/lookalike/无法判断。
  DINOv3只允许离线挑选多样困难负样本，不作为首版手机模型。
- 2026-08-31：工作重心切换到真实松动状态。新增安全的四态融合基线与真实状态复核包，19个独立
  真实参考检查点完成Codex首审：7个`LIKELY_ALIGNED`、2个`POSSIBLE_DISPLACED`、8个需补拍、
  2个不构成跨moving/fixed防松线。首版曾错误地把18/19张的锈迹、铜色或结构漆拟合成两段线；根因
  是误用依赖生成baseline的ImageGen颜色选择器，已废弃该几何建议并重建事务式v1.2，自动端点建议为0。
  未经真实端点二审和受控阈值标定时一律输出`INSUFFICIENT`。移动端接口已从丢失拒判语义的布尔值
  扩展为`ALIGNED/DISPLACED/DAMAGED_MARK/INSUFFICIENT`四态；327项Python测试、Android单测和
  Debug APK构建通过，正式真值哈希保持不变。
- 2026-08-31：`ref-09/ref-13`完成隐藏首审结论的端点二审。`ref-09`为无唯一线轴的宽涂层；
  `ref-13`端点重算夹角约1.394°但置信度0.55低于0.60门，二者均落为`INSUFFICIENT`，没有形成
  `DISPLACED`真值。新增哈希绑定二审审计器，单张历史观察不能提升为可训练状态；本轮2张已审、
  1张端点完整、0张训练可用，训练门因缺真实受控成对真值和未标定阈值继续关闭。337项Python测试
  通过、1项跳过，正式真值SHA-256保持不变。
- 2026-09-01：独立手机相机候选测试页已接入640输入YOLOv8s-P2 ONNX，并安装到华为P20 Pro
  `CLT-AL00`。唯一launcher直接进入手机后置相机，不经过眼镜/BLE/Wi-Fi流程；模型状态和候选框
  叠加显示，松动状态固定拒判。最终APK SHA-256为
  `FEBAFA80751E8CA0D753CC517CB72AEA15EF306CC6AE7D37C2AF0FE9D9B11B95`，内嵌模型哈希与外部导出
  一致。17次连续空场景推理端到端P50/P95为1915.2/2033.6ms，中位0.52 FPS，约329MB PSS；
  NNAPI和XNNPACK在该麒麟970设备上更慢，最终采用CPU 4线程。47项Android测试、337项Python测试
  通过，正式真值哈希保持不变。该结果验证手机端运行链，不代表现场准确率或松动状态能力。
- 仓库没有眼镜端 App、后台服务、SOP 引擎、登录、巡检记录、语音引导、内窥镜接入、自动化测试或 CI。
- 2026-08-25：在 Windows 中文路径下加入 `android.overridePathCheck=true` 后，
  `.\gradlew.bat assembleDebug` 构建成功；APK 大小 28,268,821 bytes，SHA-256 为
  `D9B54B8C9A2402FA3FFC83C32229397A08112610EAE4C28D522D122851263E20`。
- 2026-08-25：已加入Python数据工具测试和Android几何JUnit测试；本机测试与APK构建证据见
  `docs/validation/2026-08-25-full-image-phase1.md`。
- Phase 1 debug APK为28,325,108 bytes，SHA-256
  `F795FA260905DAF85B387EF3EE1A106C4EAB7B2593A6CFA9410EB23CB46431F6`。
- 尚未完成真实手机/眼镜验证。

## Active Work

带防松标记检查点的候选和全图审计闭环已实现，独立手机相机候选测试页也已在指定手机运行。
2026-08-31完成单类YOLO-P2候选与三分类
MobileNetV3-Small复核：在17个同源开发val场景、75个marked-point真值上，三个固定seed均保持75/75，
但单模型候选负担为22.47/17.29/23.24个/图，只有一个seed通过≤20门。固定等权几何均值为17.59个/图；
单模型权重平均挑战者为19.71个/图并保持75/75，可进入跨设备挑战但余量很小。阈值仍使用同一val选择，
sealed test未打开、当前手机实测只验证运行与性能，尚无跨车辆现场准确率证据，因此不允许表述为
生产准确率。松动状态已完成真实首审和
2个疑似点的隐藏二审；两个疑似点均因宽涂层或低分辨率落为`INSUFFICIENT`。真实数据仍没有受控
`ALIGNED/DISPLACED`成对真值，因此尚不能训练或宣称可靠状态模型。现有审计链已能阻止单张历史图
被错误提升为松动真值。

## Run

```powershell
.\gradlew.bat assembleDebug
$env:CRRC_VISION_DATA_ROOT='E:\Work\京新数智\识动hicool\中车眼镜数据资产'
.\.venv\Scripts\python.exe -m pytest ml/tests -v
```

APK 预期输出：`app/build/outputs/apk/debug/app-debug.apk`。

## Validate

```powershell
.\gradlew.bat clean assembleDebug
.\.venv\Scripts\python.exe -m pytest ml/tests -v
git status --short
```

实机验收另按 `docs/analysis/2026-08-25-demo-gap-analysis.md` 的 P0 验收门槛执行。

## Known Risks

- 当前代码来自一次性上游提交，且没有 LICENSE；代码、K900 AAR 和 native 库授权边界未确认。
- 会议没有冻结眼镜准确型号、固件、手机型号、协议版本和“rocket”品牌转写。
- 二维码可用系统解码；防松线当前P2基线AP50为0.605且尚未接入Android；万用表读数
  仍缺模型、样本、阈值和可量化验收指标。
- 当前482张来自同一约44分钟采集，缺跨车辆、跨手机、跨光照和独立测试集；没有受控“正常/松动”真值。
- 色标候选仍受锈蚀、警示贴、强光和断裂涂线影响，不能作为状态真值；整图银标训练门已PASS，但
  只覆盖单次采集和AI银标内部验证。
- 通用开放词汇模型会把波纹管、滤筒和整束管线误判为连接件，且严重漏掉小紧固件；不能作为本批
  数据的批量自动标注器。
- 参考YOLOv8s教师虽然候选质量高，但100张整图复核均未证明完整，且6张完全零检测；类别名只能从
  视觉推断、原始YAML和授权缺失，因此不能直接回写正式真值或充当生产模型。
- 眼镜照片被导入后可能不再出现在 `media.config`，失败重试和证据保全存在数据丢失风险。
- 明文 HTTP、广泛存储权限和现场影像合规尚未形成交付方案。

## Next Smallest Action

先让现场人员用已安装的手机测试页拍摄清晰、近距离、多角度检查点，形成跨手机/光照的候选误差包；
同时采集同一检查点受控`ALIGNED/DISPLACED`成对照片，标注固定侧/活动侧两段防松线端点、可见性和
维护人员确认状态，并用真实成对数据标定阈值。
状态头不得用ImageGen或单张历史图自证准确。检测分支另新增至少100–150个跨车辆、跨设备、跨光照
真实完整场景，冻结模型与阈值后再打开一次独立sealed test。通过后导出移动端模型，在指定手机连续
热机50次测端到端P50/P95和内存，并在指定手机与CY01眼镜上复验BLE连接和照片同步。

## Evidence

- `app/src/main/java/com/ar/glass/core/GlassBleService.java`：当前连接与照片同步实现。
- `app/src/main/java/com/ar/glass/vision/DefaultImageAnalyzer.java`：三个算法仍为占位。
- `docs/sources/2026-08-24-AR眼镜开发周会-逐字稿.txt`：会议需求来源。
- `docs/analysis/2026-08-25-demo-gap-analysis.md`：现状与目标差距及优先级。
- `docs/validation/2026-08-25-local-build.md`：本机路径修复、构建与测试证据。
- `docs/validation/2026-08-25-prelabel-audit.md`：60张/257候选的质量审计。
- `docs/validation/2026-08-25-training-readiness.md`：训练拒绝条件和达门后的固定训练方案。
- `docs/analysis/2026-08-25-full-image-fastener-route-v2.md`：移动端小目标路线重评与新验收门。
- `docs/validation/2026-08-25-full-image-v2-bootstrap.md`：代表帧、真值门与开放词汇试跑证据。
- `docs/validation/2026-08-28-picodet-phase-b.md`：80场景银标门、S/M训练、导出、推理和时延证据。
- `docs/validation/2026-08-28-small-object-accuracy-recovery.md`：小目标根因、P2对照、阈值与ONNX边界。
- `docs/validation/2026-08-28-high-accuracy-validation.md`：三种子严格门、M-P2挑战者、误差桶、
  密封测试和防松状态能力边界。
- `docs/validation/2026-08-29-marked-point-candidate-recall.md`：带防松标记检查点真值、全图审计、
  候选覆盖和训练拒绝结论。
- `docs/validation/2026-08-29-synthetic-marked-point-pilot.md`：ImageGen防松线来源、双层复核、
  合成全图哈希、可重复性与训练边界。
- `docs/validation/2026-08-29-synthetic-training-ablation.md`：25%合成批次公平训练、真实val结果、
  权重哈希、兼容恢复与能力边界。
- `docs/validation/2026-08-31-h1-hard-sample-review.md`：H1三轮审核、失败根因、1×/2×/4×证据链与
  真实受控状态数据切换结论。
- `docs/research/2026-08-31-yolo-accuracy-recovery/draft.md`：YOLO误差根因、两阶段准确率恢复路线、
  DINOv3边界与固定实验门。
- `docs/validation/2026-08-31-marked-point-model-recovery.md`：E1/E2/E3/E4训练、失败实验、三分类复核、
  双分数与去重后的75/75、18.35候选/图开发门及能力边界。
- `docs/validation/2026-08-31-witness-state-real-pilot.md`：真实状态首审、安全四态融合、颜色误拟合根因、
  2个疑似位移点及受控成对数据门。
- `docs/validation/2026-09-01-android-phone-live-test.md`：纯手机测试入口、APK/模型哈希、指定手机安装、
  真机性能与能力边界。
- Git外`review-packs/fastener-v2/reference-teacher-v1/ai-review-v1.json`：100图教师候选与整图复核结果。
