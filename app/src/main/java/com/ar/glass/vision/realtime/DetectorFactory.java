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
        return create(
                context,
                buildValue,
                ncnnConfidenceThreshold,
                ncnnVulkan,
                ncnnVulkanFp16,
                false);
    }

    public static FastenerDetector create(
            Context context,
            String buildValue,
            float ncnnConfidenceThreshold,
            boolean ncnnVulkan,
            boolean ncnnVulkanFp16,
            boolean markedPointVerifierEnabled) {
        return create(
                context,
                buildValue,
                ncnnConfidenceThreshold,
                ncnnVulkan,
                ncnnVulkanFp16,
                markedPointVerifierEnabled,
                false);
    }

    public static FastenerDetector create(
            Context context,
            String buildValue,
            float ncnnConfidenceThreshold,
            boolean ncnnVulkan,
            boolean ncnnVulkanFp16,
            boolean markedPointVerifierEnabled,
            boolean markedPointVerifierNnapi) {
        return create(
                context,
                buildValue,
                ncnnConfidenceThreshold,
                ncnnVulkan,
                ncnnVulkanFp16,
                markedPointVerifierEnabled,
                markedPointVerifierNnapi,
                false);
    }

    public static FastenerDetector create(
            Context context,
            String buildValue,
            float ncnnConfidenceThreshold,
            boolean ncnnVulkan,
            boolean ncnnVulkanFp16,
            boolean markedPointVerifierEnabled,
            boolean markedPointVerifierNnapi,
            boolean markedPointVerifierXnnpack) {
        DetectorBackend backend = DetectorBackend.fromBuildValue(buildValue);
        if (backend == DetectorBackend.NCNN) {
            if (markedPointVerifierEnabled) {
                return new VerifiedNcnnFastenerDetector(
                        context,
                        ncnnConfidenceThreshold,
                        ncnnVulkan,
                        ncnnVulkanFp16,
                        markedPointVerifierNnapi,
                        markedPointVerifierXnnpack);
            }
            return new NcnnFastenerDetector(
                    context, ncnnConfidenceThreshold, ncnnVulkan, ncnnVulkanFp16);
        }
        return new OnnxFastenerDetector(context);
    }
}
