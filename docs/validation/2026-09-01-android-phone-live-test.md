# Android 手机相机候选测试验收（2026-09-01）

## 结论

手机测试 APK 已安装到华为 P20 Pro（`CLT-AL00`，Android 10），唯一启动页为
`LiveInspectionActivity`。该入口直接使用手机后置摄像头和本地 ONNX，不经过眼镜、BLE、
Wi-Fi Direct 或照片同步流程。

当前 APK 可以做全图紧固件候选框测试，但不能判断防松线是否松动。界面固定显示：
`松动状态：无法判断（状态模型和真实阈值尚未就绪）`。

## 安装物与模型一致性

- 手机 ADB 序列号：`TPC7N18604005991`
- 眼镜 ADB 序列号：`1901092524001406`（本次没有安装或启动）
- APK：`app/build/outputs/apk/debug/app-debug.apk`
- APK 大小：`65,225,858 bytes`
- APK SHA-256：`FEBAFA80751E8CA0D753CC517CB72AEA15EF306CC6AE7D37C2AF0FE9D9B11B95`
- 内嵌模型：`assets/fastener-target-p2-640.onnx`
- 内嵌模型大小：`43,245,031 bytes`
- 内嵌与外部模型 SHA-256：均为
  `C50F9105FF75885BE3BA02464E6A994FA7A45FDE0B0634AEA12FAA04A6CC5B7A`
- 正式真值 SHA-256：
  `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`，保持不变。

`adb install -r` 返回 `Success`；包管理器解析出的唯一 launcher 为
`com.ar.glass/.ui.LiveInspectionActivity`。

## 真机性能

最终版本使用 ONNX Runtime CPU 4 线程。相机被遮挡的稳定空场景上采集17个连续推理结果：

| 指标 | 结果 |
|---|---:|
| 端到端 P50 / P95 | 1,915.2 / 2,033.6 ms |
| 纯模型 P50 / P95 | 1,826.0 / 1,943.4 ms |
| 预处理 P50 | 88.2 ms |
| 后处理 P50 | 0.8 ms |
| 中位更新频率 | 0.52 FPS |
| 进程 PSS | 约 329 MB |

剖析证明主要瓶颈是640输入的YOLOv8s-P2模型推理，不是相机转换或NMS。短对照中，NNAPI约
9.3秒/帧、XNNPACK约5.5秒/帧，均比CPU 4线程慢，因此没有保留。当前是实时相机预览加约两秒
一次的候选框更新，不应表述为视频级实时检测。

截图与UI层级保存在Git外：
`E:/crrc_vision_data/runs/android-live-test-2026-08-31/screen-final.png` 和
`window-final.xml`。截图时镜头被遮挡，因此候选数为0；这次实机证据证明模型加载、连续推理、
性能和安全文案，不证明真实现场检测准确率。

## 回归结果与能力边界

- Android JVM：47项通过，0失败。
- Python：337项通过，1项跳过。
- Debug APK 干净构建通过。
- 入口规范复核：`SPEC_OK`。
- 相机与性能改动代码质量复核：`QUALITY_OK`。

模型保持固定640输入、0.20候选阈值和0.45 NMS，没有为了提高帧率降低输入分辨率或候选召回。
已有同源内部基线在阈值0.20时 precision 0.641、recall 0.584，仍不是生产准确率。真实
`ALIGNED / DISPLACED`成对状态真值和现场阈值尚缺，因此松动状态必须继续拒判。
