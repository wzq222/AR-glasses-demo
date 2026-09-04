# 2026-08-25 本机构建验证

## 环境与问题

项目位于 `E:\Work\京新数智\识动hicool\中车眼镜`。首次执行 `gradlew.bat assembleDebug` 时，
Android Gradle Plugin 7.2.0 在配置阶段拒绝 Windows 非 ASCII 项目路径，尚未进入源码编译。

重复执行 `gradlew.bat tasks --quiet` 得到同一错误；工作区内其他中文路径 Android 项目均使用
`android.overridePathCheck=true`。因此在项目 `gradle.properties` 增加该官方提示的路径覆盖配置，
没有更改业务代码或依赖版本。

## 验证结果

```powershell
.\gradlew.bat tasks --quiet
.\gradlew.bat assembleDebug
.\gradlew.bat testDebugUnitTest
```

结果：

- Gradle task 列表成功加载。
- `assembleDebug`：`BUILD SUCCESSFUL`，31 个任务执行。
- `testDebugUnitTest`：`BUILD SUCCESSFUL`，但 `testDebugUnitTest NO-SOURCE`。
- APK：`app/build/outputs/apk/debug/app-debug.apk`
- APK 大小：28,268,821 bytes
- APK SHA-256：`D9B54B8C9A2402FA3FFC83C32229397A08112610EAE4C28D522D122851263E20`

会议逐字稿复制件与下载原件 SHA-256 均为
`D286500E155D2AF79A2EE7FA84BD73CE93E9353861405DFE42E7681574B4FAF7`。

## 构建警告

- Android Gradle Plugin 7.2.0 只测试到 compileSdk 32，而项目使用 compileSdk 33。
- `flatDir` 依赖仓库缺少元数据能力。
- `ksdk-release.aar` Manifest 存在重复位置权限声明。
- Java 编译报告弃用 API 与未检查操作。
- native 库未 strip，按原样打包。
- 当前没有任何单元测试源码。

这些警告不阻塞当前 debug APK 构建，但在现场试用或发布前需要逐项评估。构建成功只证明工程可编译，
不证明 BLE/WiFi、照片同步或三个识别功能在真实设备上可用。
