package com.ar.glass.vision;

import android.graphics.Bitmap;

import com.ar.glass.vision.fastener.FastenerState;

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
     * 兼容旧调用；新代码必须使用 analyzeAntiLooseState 保留拒判语义。
     *
     * @return 仅 DISPLACED 返回 true；ALIGNED、DAMAGED_MARK 和 INSUFFICIENT 均返回 false
     */
    @Deprecated
    boolean isNutLoose(Bitmap bitmap);

    /**
     * 判断防松标记所指示的相对位移，不声称螺栓预紧力或剩余扭矩。
     */
    default FastenerState analyzeAntiLooseState(Bitmap bitmap) {
        return FastenerState.INSUFFICIENT;
    }

    /**
     * 识别电压表屏幕上的数字读数。
     *
     * @param bitmap 待识别的图片
     * @return 读数字符串（如 "12.5"）；未识别到返回 null
     */
    String readMeterValue(Bitmap bitmap);
}
