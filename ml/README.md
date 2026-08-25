# 中车眼镜视觉训练工程

本目录只保存可复现的数据审计、预标注、复核、训练和导出代码。现场原图、标注结果、训练权重与
复核图片保存在Git外的私有资产根目录。

```powershell
$env:CRRC_VISION_DATA_ROOT='E:\Work\京新数智\识动hicool\中车眼镜数据资产'
.\.venv\Scripts\python.exe -m pytest ml\tests -v
```

缺少 `CRRC_VISION_DATA_ROOT` 时工具应明确失败，不能静默写入仓库或临时目录。
