package com.ar.glass.vision;

import android.content.Context;
import android.util.Log;

/**
 * YoloDetector 懒加载持有器：首次检测时才加载模型（约 1~3s），
 * 避免拖慢服务/应用启动；加载失败时 isReady()=false，检测环节自动跳过。
 */
public final class YoloDetectorHolder {

    private static final String TAG = "YoloDetectorHolder";

    private static volatile YoloDetector sDetector;
    private static volatile boolean sInitFailed = false;
    private static volatile String sInitError;
    /** 引擎加载前的预存置信度阈值（UI 滑条先于首次检测时使用） */
    private static volatile float sPendingConf = 0.25f;

    private YoloDetectorHolder() {
    }

    public static YoloDetector get(Context context) {
        if (sDetector == null) {
            synchronized (YoloDetectorHolder.class) {
                if (sDetector == null && !sInitFailed) {
                    try {
                        sDetector = YoloDetector.get(context.getApplicationContext());
                        if (sDetector != null && sDetector.isReady()) {
                            sDetector.setConfThreshold(sPendingConf);
                            sInitError = null;
                        } else {
                            sInitFailed = true;
                            sInitError = "ONNX 会话创建失败（详见 logcat YoloDetector）";
                        }
                    } catch (Throwable e) {
                        sInitFailed = true;
                        sInitError = e.getClass().getSimpleName() + ": " + e.getMessage();
                        Log.e(TAG, "YoloDetector init failed", e);
                    }
                }
            }
        }
        return sDetector;
    }

    /** 引擎是否已就绪可用 */
    public static boolean isReady() {
        YoloDetector d = sDetector;
        return d != null && d.isReady();
    }

    /** 首次加载失败的原因（用于界面提示；成功或未尝试过返回 null） */
    public static String getInitError() {
        return sInitError;
    }

    /** 允许下一次 get() 重试加载（如清理缓存后） */
    public static void reset() {
        synchronized (YoloDetectorHolder.class) {
            sInitFailed = false;
            sInitError = null;
        }
    }

    /** 设置置信度阈值：引擎未加载时先暂存，加载后自动应用 */
    public static void setConfThreshold(float v) {
        sPendingConf = v;
        YoloDetector d = sDetector;
        if (d != null && d.isReady()) d.setConfThreshold(v);
    }
}
