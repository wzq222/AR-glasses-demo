package com.ar.glass.vision.realtime;

import java.util.Arrays;

/** Measured anti-loosening witness-line geometry plus conservative review routing. */
public final class WitnessStateEstimate {
    public static final float REVIEW_THRESHOLD_DEGREES = 3f;
    public static final float HIGH_SUSPICION_THRESHOLD_DEGREES = 15f;
    public static final float EXPERIMENTAL_ERROR_DEGREES = 6.3f;

    private final WitnessTriage triage;
    private final float angleDegrees;
    private final float lowerDegrees;
    private final float upperDegrees;
    private final float[] normalizedPoints;
    private final String reason;
    private final double inferenceMillis;

    private WitnessStateEstimate(
            WitnessTriage triage,
            float angleDegrees,
            float lowerDegrees,
            float upperDegrees,
            float[] normalizedPoints,
            String reason,
            double inferenceMillis) {
        this.triage = triage;
        this.angleDegrees = angleDegrees;
        this.lowerDegrees = lowerDegrees;
        this.upperDegrees = upperDegrees;
        this.normalizedPoints = normalizedPoints == null ? null : normalizedPoints.clone();
        this.reason = reason;
        this.inferenceMillis = inferenceMillis;
    }

    public static WitnessStateEstimate measured(
            float angleDegrees, float[] normalizedPoints, double inferenceMillis) {
        if (!Float.isFinite(angleDegrees) || angleDegrees < 0f || angleDegrees > 90f
                || normalizedPoints == null || normalizedPoints.length != 8
                || !Double.isFinite(inferenceMillis) || inferenceMillis < 0.0) {
            throw new IllegalArgumentException("witness estimate values are invalid");
        }
        for (float coordinate : normalizedPoints) {
            if (!Float.isFinite(coordinate) || coordinate < 0f || coordinate > 1f) {
                throw new IllegalArgumentException("witness point is outside normalized bounds");
            }
        }
        WitnessTriage triage;
        String reason;
        if (angleDegrees <= REVIEW_THRESHOLD_DEGREES) {
            triage = WitnessTriage.LIKELY_ALIGNED;
            reason = "ANGLE_AT_OR_BELOW_REVIEW_THRESHOLD";
        } else if (angleDegrees < HIGH_SUSPICION_THRESHOLD_DEGREES) {
            triage = WitnessTriage.POSSIBLE_DISPLACED;
            reason = "ANGLE_REVIEW_REQUIRED";
        } else {
            triage = WitnessTriage.HIGH_SUSPICION;
            reason = "SECOND_VIEW_CONFIRMATION_REQUIRED";
        }
        return new WitnessStateEstimate(
                triage,
                angleDegrees,
                Math.max(0f, angleDegrees - EXPERIMENTAL_ERROR_DEGREES),
                Math.min(90f, angleDegrees + EXPERIMENTAL_ERROR_DEGREES),
                normalizedPoints,
                reason,
                inferenceMillis);
    }

    public static WitnessStateEstimate insufficient(String reason) {
        return new WitnessStateEstimate(
                WitnessTriage.INSUFFICIENT,
                Float.NaN,
                Float.NaN,
                Float.NaN,
                null,
                reason == null ? "MEASUREMENT_UNAVAILABLE" : reason,
                0.0);
    }

    public boolean isMeasured() { return triage != WitnessTriage.INSUFFICIENT; }
    public WitnessTriage getTriage() { return triage; }
    public float getAngleDegrees() { return angleDegrees; }
    public float getLowerDegrees() { return lowerDegrees; }
    public float getUpperDegrees() { return upperDegrees; }
    public float[] getNormalizedPoints() {
        return normalizedPoints == null ? null : Arrays.copyOf(normalizedPoints, normalizedPoints.length);
    }
    public String getReason() { return reason; }
    public double getInferenceMillis() { return inferenceMillis; }
}
