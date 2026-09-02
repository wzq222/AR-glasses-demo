package com.ar.glass.vision;

import android.content.Context;
import android.graphics.Bitmap;

import com.ar.glass.BuildConfig;
import com.ar.glass.vision.realtime.Detection;
import com.ar.glass.vision.realtime.DetectorFactory;
import com.ar.glass.vision.realtime.FastenerDetector;
import com.ar.glass.vision.realtime.OnnxWitnessStateEstimator;
import com.ar.glass.vision.realtime.OnnxFastenerDetector;
import com.ar.glass.vision.realtime.WitnessRoi;
import com.ar.glass.vision.realtime.WitnessStateEstimate;
import com.ar.glass.vision.realtime.WitnessTriage;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Product-facing singleton for the validated marked-point proposal + verifier chain. */
public final class MarkedPointDetectorHolder {
    private static volatile FastenerDetector detector;
    private static volatile String initializationError;
    private static volatile OnnxWitnessStateEstimator stateEstimator;
    private static volatile String stateInitializationError;
    private static final float MINIMUM_STATE_ROI_SIDE_PIXELS = 32f;

    private MarkedPointDetectorHolder() {}

    private static FastenerDetector get(Context context) {
        if (detector == null && initializationError == null) {
            synchronized (MarkedPointDetectorHolder.class) {
                if (detector == null && initializationError == null) {
                    FastenerDetector candidate = DetectorFactory.create(
                            context.getApplicationContext(),
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

    public static boolean isReady(Context context) {
        return get(context) != null;
    }

    public static String getInitializationError() {
        return initializationError;
    }

    private static OnnxWitnessStateEstimator getStateEstimator(Context context) {
        if (stateEstimator == null && stateInitializationError == null) {
            synchronized (MarkedPointDetectorHolder.class) {
                if (stateEstimator == null && stateInitializationError == null) {
                    OnnxWitnessStateEstimator candidate = new OnnxWitnessStateEstimator(
                            context.getApplicationContext());
                    if (candidate.isReady()) {
                        stateEstimator = candidate;
                    } else {
                        stateInitializationError = candidate.getInitializationError();
                        candidate.close();
                    }
                }
            }
        }
        return stateEstimator;
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
            return new Result(
                    Collections.emptyList(), Collections.emptyList(), raw.getLatencyMillis(), 0.0);
        }
        List<YoloDetector.Detection> mapped = new ArrayList<>();
        List<Assessment> assessments = new ArrayList<>();
        OnnxWitnessStateEstimator witnessEstimator = getStateEstimator(context);
        double stateLatencyMillis = 0.0;
        int index = 0;
        for (Detection hit : raw.getDetections()) {
            index++;
            WitnessRoi roi = null;
            WitnessStateEstimate estimate;
            float side = Math.max(hit.getRight() - hit.getLeft(), hit.getBottom() - hit.getTop());
            if (side < MINIMUM_STATE_ROI_SIDE_PIXELS) {
                estimate = WitnessStateEstimate.insufficient("TARGET_TOO_SMALL_RECAPTURE_CLOSER");
            } else if (witnessEstimator == null) {
                estimate = WitnessStateEstimate.insufficient(
                        stateInitializationError == null
                                ? "STATE_MODEL_UNAVAILABLE" : stateInitializationError);
            } else {
                try {
                    roi = WitnessRoi.fromDetection(hit, width, height);
                    estimate = witnessEstimator.estimate(bitmap, roi);
                    stateLatencyMillis += estimate.getInferenceMillis();
                } catch (Exception | LinkageError error) {
                    estimate = WitnessStateEstimate.insufficient(
                            error.getMessage() == null
                                    ? "STATE_MEASUREMENT_FAILED" : error.getMessage());
                }
            }
            float[] fullImagePoints = roi == null || !estimate.isMeasured()
                    ? null : mapPointsToImage(estimate.getNormalizedPoints(), roi, width, height);
            String label = "防松点" + index + "·" + triageLabel(estimate.getTriage());
            mapped.add(new YoloDetector.Detection(
                    clamp01(hit.getLeft() / width),
                    clamp01(hit.getTop() / height),
                    clamp01(hit.getRight() / width),
                    clamp01(hit.getBottom() / height),
                    hit.getConfidence(),
                    hit.getClassId(),
                    label,
                    estimate.getTriage().name(),
                    estimate.getAngleDegrees(),
                    fullImagePoints));
            assessments.add(new Assessment(
                    index,
                    clamp01(hit.getLeft() / width),
                    clamp01(hit.getTop() / height),
                    clamp01(hit.getRight() / width),
                    clamp01(hit.getBottom() / height),
                    hit.getConfidence(),
                    estimate));
        }
        return new Result(
                Collections.unmodifiableList(mapped),
                Collections.unmodifiableList(assessments),
                raw.getLatencyMillis(),
                stateLatencyMillis);
    }

    private static float[] mapPointsToImage(
            float[] normalizedPoints, WitnessRoi roi, int imageWidth, int imageHeight) {
        float[] mapped = new float[8];
        for (int index = 0; index < 4; index++) {
            mapped[index * 2] = clamp01(
                    (roi.getLeft() + normalizedPoints[index * 2] * roi.getWidth()) / imageWidth);
            mapped[index * 2 + 1] = clamp01(
                    (roi.getTop() + normalizedPoints[index * 2 + 1] * roi.getHeight()) / imageHeight);
        }
        return mapped;
    }

    private static String triageLabel(WitnessTriage triage) {
        if (triage == WitnessTriage.LIKELY_ALIGNED) return "正常倾向";
        if (triage == WitnessTriage.POSSIBLE_DISPLACED) return "疑似错位";
        if (triage == WitnessTriage.HIGH_SUSPICION) return "高疑似·待确认";
        return "需近拍";
    }

    private static float clamp01(float value) {
        return Math.max(0f, Math.min(1f, value));
    }

    public static final class Result {
        public final List<YoloDetector.Detection> detections;
        public final List<Assessment> assessments;
        public final double latencyMillis;
        public final double detectorLatencyMillis;
        public final double stateLatencyMillis;

        Result(
                List<YoloDetector.Detection> detections,
                List<Assessment> assessments,
                double detectorLatencyMillis,
                double stateLatencyMillis) {
            this.detections = detections;
            this.assessments = assessments;
            this.detectorLatencyMillis = detectorLatencyMillis;
            this.stateLatencyMillis = stateLatencyMillis;
            this.latencyMillis = detectorLatencyMillis + stateLatencyMillis;
        }
    }

    public static final class Assessment {
        public final int index;
        public final float left;
        public final float top;
        public final float right;
        public final float bottom;
        public final float detectionConfidence;
        public final WitnessStateEstimate estimate;

        Assessment(
                int index,
                float left,
                float top,
                float right,
                float bottom,
                float detectionConfidence,
                WitnessStateEstimate estimate) {
            this.index = index;
            this.left = left;
            this.top = top;
            this.right = right;
            this.bottom = bottom;
            this.detectionConfidence = detectionConfidence;
            this.estimate = estimate;
        }
    }
}
