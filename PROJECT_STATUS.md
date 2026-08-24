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
- 仓库没有眼镜端 App、后台服务、SOP 引擎、登录、巡检记录、语音引导、内窥镜接入、自动化测试或 CI。
- 2026-08-25：在 Windows 中文路径下加入 `android.overridePathCheck=true` 后，
  `.\gradlew.bat assembleDebug` 构建成功；APK 大小 28,268,821 bytes，SHA-256 为
  `D9B54B8C9A2402FA3FFC83C32229397A08112610EAE4C28D522D122851263E20`。
- 2026-08-25：`.\gradlew.bat testDebugUnitTest` 成功，但结果为 `NO-SOURCE`，证明当前仓库没有单元测试，
  不能视为功能测试通过。
- 尚未完成真实手机/眼镜验证。

## Active Work

已完成项目初始化、会议原始材料归档、需求提取、代码缺口分析和两周 MVP 设计。尚未开始功能实现。

## Run

```powershell
.\gradlew.bat assembleDebug
```

APK 预期输出：`app/build/outputs/apk/debug/app-debug.apk`。

## Validate

```powershell
.\gradlew.bat clean assembleDebug
git status --short
```

实机验收另按 `docs/analysis/2026-08-25-demo-gap-analysis.md` 的 P0 验收门槛执行。

## Known Risks

- 当前代码来自一次性上游提交，且没有 LICENSE；代码、K900 AAR 和 native 库授权边界未确认。
- 会议没有冻结眼镜准确型号、固件、手机型号、协议版本和“rocket”品牌转写。
- 三种识别算法缺模型、样本、阈值和可量化验收指标。
- 眼镜照片被导入后可能不再出现在 `media.config`，失败重试和证据保全存在数据丢失风险。
- 明文 HTTP、广泛存储权限和现场影像合规尚未形成交付方案。

## Next Smallest Action

在指定 Android 手机与指定 CY01 眼镜上完成基线构建、安装、BLE 连接和照片同步复验，记录设备/固件
矩阵与失败日志；没有这一步，不进入算法集成。

## Evidence

- `app/src/main/java/com/ar/glass/core/GlassBleService.java`：当前连接与照片同步实现。
- `app/src/main/java/com/ar/glass/vision/DefaultImageAnalyzer.java`：三个算法仍为占位。
- `docs/sources/2026-08-24-AR眼镜开发周会-逐字稿.txt`：会议需求来源。
- `docs/analysis/2026-08-25-demo-gap-analysis.md`：现状与目标差距及优先级。
- `docs/validation/2026-08-25-local-build.md`：本机路径修复、构建与测试证据。
