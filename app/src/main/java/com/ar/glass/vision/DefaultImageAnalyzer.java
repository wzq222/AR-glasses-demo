package com.ar.glass.vision;

import android.graphics.Bitmap;

import com.ar.glass.vision.cloud.MeterCloudOcr;

/**
 * 识别接口的默认实现。
 *
 * 「万用表读数识别」：
 * - readMeterValue：接入火山引擎豆包视觉大模型 API（云端 OCR）
 *   （需在 gradle.properties 配置 ARK_API_KEY，详见 MeterCloudOcr）
 *
 * 其余两个能力暂为占位（预留）：
 * - 二维码：ZXing 或 ML Kit Barcode Scanning
 * - 防松线错位：自定义图像处理 / 目标检测模型
 */
public class DefaultImageAnalyzer implements ImageAnalyzer {

    @Override
    public String decodeQrCode(Bitmap bitmap) {
        // TODO 接入二维码识别
        return null;
    }

    @Override
    public boolean isNutLoose(Bitmap bitmap) {
        // TODO 接入防松线错位检测
        return false;
    }

    @Override
    public String readMeterValue(Bitmap bitmap) {
        MeterReading r = readMeter(bitmap);
        if (r == null) {
            return null;
        }
        String text = r.getDisplayText();
        return text.isEmpty() ? null : text;
    }

    @Override
    public MeterReading readMeter(Bitmap bitmap) {
        // 云端识别（读数 + 单位 + 挡位 + 异常），需后台线程调用
        return MeterCloudOcr.recognizeMeter(bitmap);
    }
}
