# APK 内置案例图库验证

## 结果

Debug APK 已把用户提供的 7 张案例图片直接打包到 `assets/builtin_gallery`。App 打开原图库时，
会在后台安全释放案例，并将内置案例与现有眼镜同步/用户导入图片合并展示；SOP 的“从原图库选择”
入口可直接选择这些文件。检测模型、阈值和后台流程均未修改。

现场图片仍在 Git 外。构建通过 `CRRC_BUILTIN_GALLERY_DIR` 读取
`E:\Work\京新数智\识动hicool\中车眼镜数据资产\app-original-gallery`，只复制图片到 Gradle
生成目录和最终 APK，Git 历史不包含案例原图。

## 内置内容

- `fastener/LOCK-REAL-01.jpg`
- `meter/METER-04_zero_000.jpg`
- `meter/METER-13_dual_7.2_230.png`
- `meter/METER-21_26.62V_DC.png`
- `meter/METER-23_233.5V_AC.jpg`
- `meter/METER-29_main_-0.04A.jpg`
- `qr/QR-REAL-01.jpg`

APK 内逐文件 SHA-256 与 Git 外源图片全部一致。

## 验证

- Android JVM：78 tests，0 failures，0 errors，0 skipped。
- `clean :app:testDebugUnitTest :app:assembleDebug`：BUILD SUCCESSFUL。
- APK 内容：二维码 1、防松线 1、万用表 5，共 7 张；非图片 manifest 未打包。
- APK 包信息：`com.ar.glass`，version `1.0.0`，minSdk 22，targetSdk 33。
- APK 大小：111,263,539 bytes。
- APK SHA-256：`EC9FF9F054605EF94D474C28672EF2EFE3864F5584027222172BB33A676AA8D0`。
- 交付文件：
  `E:\Work\京新数智\识动hicool\中车眼镜数据资产\deliverables\中车眼镜-内置案例-v1-debug.apk`。
- ASCII 兼容副本：
  `E:\crrc_vision_data\deliverables\crrc-glasses-builtin-cases-v1-debug.apk`，哈希相同。
- 本轮最终检查时 ADB 无在线设备，因此没有覆盖安装或真机 UI 冒烟；APK 构建和内容验证已完成。
- formal truth SHA-256 保持
  `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`。

## 能力边界

内置图片用于离线重复测试 SOP 和现有算法，不改变此前准确性结论。尤其
`LOCK-REAL-01.jpg` 仍是高召回/重框困难样本，不能因为被内置到 APK 就视为检测或松动状态已通过。
