package com.ar.glass.vision.realtime;

import android.content.Context;
import android.graphics.Bitmap;

public final class VerifiedNcnnFastenerDetector implements FastenerDetector {
    private final NcnnFastenerDetector detector;
    private final MarkedPointOnnxVerifier verifier;
    private boolean closed;

    public VerifiedNcnnFastenerDetector(
            Context context,
            float confidenceThreshold,
            boolean useVulkan,
            boolean useVulkanFp16) {
        this(context, confidenceThreshold, useVulkan, useVulkanFp16, false);
    }

    public VerifiedNcnnFastenerDetector(
            Context context,
            float confidenceThreshold,
            boolean useVulkan,
            boolean useVulkanFp16,
            boolean useVerifierNnapi) {
        this(
                context,
                confidenceThreshold,
                useVulkan,
                useVulkanFp16,
                useVerifierNnapi,
                false);
    }

    public VerifiedNcnnFastenerDetector(
            Context context,
            float confidenceThreshold,
            boolean useVulkan,
            boolean useVulkanFp16,
            boolean useVerifierNnapi,
            boolean useVerifierXnnpack) {
        detector = new NcnnFastenerDetector(
                context, confidenceThreshold, useVulkan, useVulkanFp16);
        verifier = new MarkedPointOnnxVerifier(
                context, useVerifierNnapi, useVerifierXnnpack);
    }

    @Override
    public synchronized boolean isReady() {
        return !closed && detector.isReady() && verifier.isReady();
    }

    @Override
    public synchronized String getInitializationError() {
        if (detector.getInitializationError() != null) {
            return detector.getInitializationError();
        }
        return verifier.getInitializationError();
    }

    @Override
    public synchronized OnnxFastenerDetector.DetectionResult detect(Bitmap bitmap)
            throws Exception {
        if (!isReady()) {
            throw new IllegalStateException(getInitializationError());
        }
        long started = System.nanoTime();
        OnnxFastenerDetector.DetectionResult proposals = detector.detect(bitmap);
        MarkedPointOnnxVerifier.VerificationResult verified = verifier.verify(
                bitmap, proposals.getDetections());
        long completed = System.nanoTime();
        return new OnnxFastenerDetector.DetectionResult(
                verified.getDetections(),
                proposals.getOriginalWidth(),
                proposals.getOriginalHeight(),
                (completed - started) / 1_000_000.0,
                proposals.getPreprocessMillis() + verified.getPreprocessMillis(),
                proposals.getInferenceMillis() + verified.getInferenceMillis(),
                proposals.getPostprocessMillis() + verified.getPostprocessMillis(),
                proposals.getTransform());
    }

    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        verifier.close();
        detector.close();
    }
}
