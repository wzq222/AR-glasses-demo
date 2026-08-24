package com.ar.glass.vision;

import android.graphics.Bitmap;

/**
 * 识别接口的占位实现（预留）。
 *
 * 当前三个方法均未接入真实算法，直接返回默认值。
 * 后续接入方向：
 * - 二维码：ZXing 或 ML Kit Barcode Scanning
 * - 防松线错位：自定义图像处理 / 目标检测模型
 * - 电压表数字：ML Kit Text Recognition 或 PaddleOCR
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
        // TODO 接入电压表数字 OCR
        return null;
    }
}
