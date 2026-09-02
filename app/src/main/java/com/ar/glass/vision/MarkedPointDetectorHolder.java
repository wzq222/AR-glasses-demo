package com.ar.glass.vision;

import android.content.Context;
import android.graphics.Bitmap;

import com.ar.glass.BuildConfig;
import com.ar.glass.vision.realtime.Detection;
import com.ar.glass.vision.realtime.DetectorFactory;
import com.ar.glass.vision.realtime.FastenerDetector;
import com.ar.glass.vision.realtime.OnnxFastenerDetector;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Product-facing singleton for the validated marked-point proposal + verifier chain. */
public final class MarkedPointDetectorHolder {
    private static final String CUSTOM_MODEL_FILE = "custom_model.onnx";

    private static volatile FastenerDetector detector;
    private static volatile String initializationError;

    private MarkedPointDetectorHolder() {}

    /** 用户更换的自定义模型文件（存在即启用，删除即恢复内置模型） */
    public static File getCustomModelFile(Context context) {
        return new File(context.getFilesDir(), CUSTOM_MODEL_FILE);
    }

    public static boolean isCustomModelActive(Context context) {
        return getCustomModelFile(context).exists();
    }

    private static FastenerDetector get(Context context) {
        if (detector == null && initializationError == null) {
            synchronized (MarkedPointDetectorHolder.class) {
                if (detector == null && initializationError == null) {
                    Context app = context.getApplicationContext();
                    File custom = getCustomModelFile(app);
                    FastenerDetector candidate = custom.exists()
                            ? new OnnxFastenerDetector(custom)
                            : DetectorFactory.create(
                                    app,
                                    BuildConfig.DETECTOR_BACKEND,
                                    BuildConfig.NCNN_CONFIDENCE_THRESHOLD,
                                    BuildConfig.NCNN_VULKAN,
                                    BuildConfig.NCNN_VULKAN_FP16,
                                    BuildConfig.MARKED_POINT_VERIFIER_ENABLED,
                                    BuildConfig.MARKED_POINT_VERIFIER_NNAPI,
                                    BuildConfig.MARKED_POINT_VERIFIER_XNNPACK);
                    if (candidate.isReady()) {
                        detector = candidate;
                    } else {
                        initializationError = candidate.getInitializationError();
                        candidate.close();
                    }
                }
            }
        }
        return detector;
    }

    /**
     * 重新加载检测器（更换/恢复模型后调用）。
     * @return 错误信息；null 表示加载成功
     */
    public static String reload(Context context) {
        synchronized (MarkedPointDetectorHolder.class) {
            if (detector != null) {
                detector.close();
                detector = null;
            }
            initializationError = null;
        }
        return isReady(context.getApplicationContext())
                ? null
                : (initializationError != null ? initializationError : "模型加载失败");
    }

    /** 删除自定义模型并恢复内置模型。 @return 错误信息；null 表示成功 */
    public static String restoreDefault(Context context) {
        File custom = getCustomModelFile(context);
        if (custom.exists() && !custom.delete()) {
            return "无法删除自定义模型文件";
        }
        return reload(context);
    }

    public static boolean isReady(Context context) {
        return get(context) != null;
    }

    public static String getInitializationError() {
        return initializationError;
    }

    public static Result detect(Context context, Bitmap bitmap) throws Exception {
        FastenerDetector active = get(context);
        if (active == null) {
            throw new IllegalStateException(initializationError == null
                    ? "防松标记模型未就绪" : initializationError);
        }
        OnnxFastenerDetector.DetectionResult raw = active.detect(bitmap);
        int width = raw.getOriginalWidth();
        int height = raw.getOriginalHeight();
        if (width <= 0 || height <= 0) {
            return new Result(Collections.emptyList(), raw.getLatencyMillis());
        }
        List<YoloDetector.Detection> mapped = new ArrayList<>();
        for (Detection hit : raw.getDetections()) {
            mapped.add(new YoloDetector.Detection(
                    clamp01(hit.getLeft() / width),
                    clamp01(hit.getTop() / height),
                    clamp01(hit.getRight() / width),
                    clamp01(hit.getBottom() / height),
                    hit.getConfidence(),
                    hit.getClassId(),
                    "防松标记"));
        }
        return new Result(Collections.unmodifiableList(mapped), raw.getLatencyMillis());
    }

    private static float clamp01(float value) {
        return Math.max(0f, Math.min(1f, value));
    }

    public static final class Result {
        public final List<YoloDetector.Detection> detections;
        public final double latencyMillis;

        Result(List<YoloDetector.Detection> detections, double latencyMillis) {
            this.detections = detections;
            this.latencyMillis = latencyMillis;
        }
    }
}
