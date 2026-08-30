package com.ar.glass.vision;

/**
 * 识别引擎入口。
 *
 * 默认使用 {@link DefaultImageAnalyzer}（二维码识别已接入 ML Kit）。
 * 可通过 {@link #set(ImageAnalyzer)} 替换实现，
 * UI 层统一通过 {@link #get()} 调用，无需改动调用方。
 */
public final class Vision {

    private static volatile ImageAnalyzer sInstance;

    private Vision() {
    }

    public static ImageAnalyzer get() {
        if (sInstance == null) {
            synchronized (Vision.class) {
                if (sInstance == null) {
                    sInstance = new DefaultImageAnalyzer();
                }
            }
        }
        return sInstance;
    }

    public static void set(ImageAnalyzer analyzer) {
        sInstance = analyzer;
    }
}
