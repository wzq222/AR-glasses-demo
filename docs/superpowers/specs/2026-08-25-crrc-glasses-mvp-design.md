# 中车眼镜两周 MVP 设计

## 目标与范围

本设计把会议目标压缩为一个可验证的闭环：在固定 CY01 眼镜和 Android 手机上，执行二维码、
防松线、万用表三步巡检，保存结构化结果和证据照片。算法允许先调用服务器；没有现场证据时不宣称
离线或端侧能力。

本轮不实现通用 SOP 编辑器、完整多用户后台、内窥镜、3D Gaussian、跨品牌眼镜适配。

## 方案选择

采用“保留上游连接层 + 新增独立作业域”的渐进方案：不重写 1382 行的 `GlassBleService`，先把
照片来源统一成一个媒体入口，再让固定 SOP、分析适配器和证据记录围绕该入口工作。这样能最大限度
复用已经写好的 BLE/WiFi 逻辑，也能用手机相机/相册绕开硬件阻塞。

## 组件边界

| 组件 | 责任 | 依赖 |
|---|---|---|
| `MediaSource` | 从眼镜同步、手机拍照或相册返回本地图片 URI | Android 媒体与现有 BLE 服务 |
| `InspectionStep` | 定义 `QR`、`NUT_LINE`、`METER` 三个固定步骤 | 无 |
| `InspectionSession` | 驱动步骤状态、重拍、确认、完成/失败 | `MediaSource`、分析器 |
| `ImageAnalyzer` 适配层 | 输出统一结构化结果，不关心图片来源 | 本地库或服务器 API |
| `EvidenceRepository` | 以 session 为单位保存 JSON 与证据 URI | App 私有存储 |
| `InspectionActivity` | 显示当前步骤、采集入口、结果确认和错误恢复 | 上述域组件 |

现有 `Vision` 静态单例不直接承担网络、状态和存储职责；它应退化为分析器装配入口，避免把流程继续
塞进 `MainActivity` 或 `GlassBleService`。

## 统一结果契约

每个步骤返回：

```json
{
  "step": "METER",
  "status": "SUCCEEDED",
  "value": "24.6",
  "unit": "V",
  "confidence": 0.94,
  "requiresHumanReview": false,
  "capturedAt": "2026-08-25T10:30:00+08:00",
  "imageUri": "content://...",
  "analyzerVersion": "meter-server-v1",
  "errorCode": null
}
```

二维码的 `value` 保存点位码；防松线的 `value` 仅取 `ALIGNED`、`MISALIGNED`、`UNCERTAIN`；
万用表拆分 `value` 与 `unit`。低置信度必须进入人工确认，不把算法猜测写成确定结论。

## 状态与错误处理

会话状态固定为 `READY → CAPTURING → ANALYZING → REVIEWING → SAVED`。任一步可进入
`RETRYABLE_ERROR`；连接、超时、模糊、无目标和服务不可用分别使用稳定错误码。原图在成功保存或
用户明确放弃前不能删除。退出会话时，无论成功、取消或异常，都调用现有退出导入模式逻辑。

## 数据与安全

- 演示版先写 App 私有目录中的 JSON，不依赖后台才能完成一轮巡检。
- 服务器算法仅上传当前步骤图片，不携带姓名、设备唯一标识或无关 EXIF。
- API 地址与凭证来自未入库配置；日志不得输出 token 或完整现场图片地址。
- 每条记录包含硬件/固件与算法版本，方便复现实验。

## 验证设计

- 单元测试覆盖 SOP 状态转换、失败重试、低置信度人工确认和 JSON 序列化。
- 仪器测试覆盖相册导入、三步完成、取消时退出导入模式。
- 三类样本各建立受控集，分正常、异常、暗光、反光、模糊；准确率与失败样本随模型版本保存。
- 实机验证使用固定手机/眼镜/固件组合，连续执行连接、同步和三步流程。

## 成功标准

用户可以不改代码、不改文件，连续完成三轮三步巡检；每轮生成一份能对应三张证据图的 JSON；
任一步失败可重拍或人工确认；结束后眼镜恢复拍照。满足这些条件才称为“巡检 MVP”。
