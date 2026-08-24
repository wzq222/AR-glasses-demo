package com.ar.glass.vision;

import android.graphics.Bitmap;

/**
 * 图像识别接口（预留）。
 *
 * 定义三个识别能力，后续接入具体算法/模型实现：
 * 1. 二维码识别
 * 2. 防松线错位检测（螺母松动）
 * 3. 电压表屏幕数字识别
 */
public interface ImageAnalyzer {

    /**
     * 识别图片中的二维码，返回二维码内容。
     *
     * @param bitmap 待识别的图片
     * @return 二维码解析出的文本；未识别到返回 null
     */
    String decodeQrCode(Bitmap bitmap);

    /**
     * 检测图片中螺母防松线是否错位（即螺母是否松动）。
     *
     * @param bitmap 待检测的图片
     * @return true 表示防松线错位（螺母松动）；false 表示未松动或未检测到
     */
    boolean isNutLoose(Bitmap bitmap);

    /**
     * 识别电压表屏幕上的数字读数。
     *
     * @param bitmap 待识别的图片
     * @return 读数字符串（如 "12.5"）；未识别到返回 null
     */
    String readMeterValue(Bitmap bitmap);
}
