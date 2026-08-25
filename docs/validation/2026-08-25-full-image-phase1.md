# 中车眼镜全图防松线 Phase 1 验证（2026-08-25）

## 本轮交付边界

已交付可复现的数据审计/预标注/复核/训练门工具，以及Android全图多目标结果和几何判定核心。检测模型
没有训练，也没有接入APK：候选标签质量未达预先规定门槛，强行训练会把锈蚀和警示贴学成紧固件。

## 数据证据

- 私有资产根：`E:/Work/京新数智/识动hicool/中车眼镜数据资产`（Git外）。
- 原图482张，2000×1500，精确重复0；177个场景组。
- 训练/内部验证424/58，组级泄漏0。
- HSV候选v2：1,993个；分层复核60张、257候选。
- 复核：接受151、拒绝97、需人工9；候选精确率60.89%；训练门FAIL。

## 2026-08-25最终验证命令

```powershell
$env:CRRC_VISION_DATA_ROOT='E:\Work\京新数智\识动hicool\中车眼镜数据资产'
.\.venv\Scripts\python.exe -m pytest ml/tests -v
.\gradlew.bat testDebugUnitTest assembleDebug
git diff --check
git ls-files | Select-String -Pattern '\.(jpg|jpeg|png|pth|onnx|param|bin)$'
```

结果：

- Python：21 tests collected，21 passed。
- Android：`testDebugUnitTest`与`assembleDebug`成功；5个几何JUnit用例通过。
- `git diff --check`退出码0。
- Git跟踪的二进制图片只有原有Android launcher图标；没有现场JPG、模型权重或导出文件进入Git。
- Debug APK：28,325,108 bytes；SHA-256
  `F795FA260905DAF85B387EF3EE1A106C4EAB7B2593A6CFA9410EB23CB46431F6`。

Gradle仍报告原项目已有的Android Gradle Plugin 7.2/compileSdk 33兼容性提示、`flatDir`提示和Gradle 8弃用
提示；本轮没有升级构建链，避免把算法底座工作扩成无关迁移。

## 已验证与未验证

代码层已验证：中文路径图片读取、清单哈希、场景拆分不泄漏、颜色候选边界、COCO合法性、复核决定完整性、
训练拒绝策略、无向夹角/低置信度/未标定阈值几何行为、APK编译打包。

尚未验证：真实手机耗时、ncnn/ONNX推理、不同车型与光照泛化、松动/正常分类准确率、CY01眼镜端到端流程。
没有这些证据前，项目状态只能称为“数据和移动端算法底座”，不能称为“全图检测已完成”。

## 下一门槛

人工确认80—120个场景组的紧固件框和色标端点，并另采受控松动/正常对；达到标签精确率门后训练
D-FINE-N候选模型，导出ONNX并与RTMDet-tiny-P2/ncnn手机实测对比。最终技术选型以目标手机上的召回率、
P95耗时、内存和包体为准，而不是以论文模型大小决定。
