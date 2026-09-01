package com.ar.glass.vision.realtime;

import android.content.Context;

public final class DetectorFactory {
    private DetectorFactory() {
    }

    public static FastenerDetector create(Context context, String buildValue) {
        return create(
                context,
                buildValue,
                YoloPostprocessor.DEFAULT_CONFIDENCE_THRESHOLD);
    }

    public static FastenerDetector create(
            Context context, String buildValue, float ncnnConfidenceThreshold) {
        DetectorBackend backend = DetectorBackend.fromBuildValue(buildValue);
        if (backend == DetectorBackend.NCNN) {
            return new NcnnFastenerDetector(context, ncnnConfidenceThreshold);
        }
        return new OnnxFastenerDetector(context);
    }
}
