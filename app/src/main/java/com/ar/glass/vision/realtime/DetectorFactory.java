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
        return create(context, buildValue, ncnnConfidenceThreshold, false, false);
    }

    public static FastenerDetector create(
            Context context,
            String buildValue,
            float ncnnConfidenceThreshold,
            boolean ncnnVulkan) {
        return create(context, buildValue, ncnnConfidenceThreshold, ncnnVulkan, false);
    }

    public static FastenerDetector create(
            Context context,
            String buildValue,
            float ncnnConfidenceThreshold,
            boolean ncnnVulkan,
            boolean ncnnVulkanFp16) {
        DetectorBackend backend = DetectorBackend.fromBuildValue(buildValue);
        if (backend == DetectorBackend.NCNN) {
            return new NcnnFastenerDetector(
                    context, ncnnConfidenceThreshold, ncnnVulkan, ncnnVulkanFp16);
        }
        return new OnnxFastenerDetector(context);
    }
}
