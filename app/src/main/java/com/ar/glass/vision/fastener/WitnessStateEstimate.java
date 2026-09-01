package com.ar.glass.vision.fastener;

public final class WitnessStateEstimate {
    public static final float EXPERIMENTAL_ERROR_DEGREES = 6.3f;

    private final AngleInterval angle;
    private final WitnessReviewHint reviewHint;
    private final String reviewReason;
    private final double inferenceMillis;

    private WitnessStateEstimate(
            AngleInterval angle,
            WitnessReviewHint reviewHint,
            String reviewReason,
            double inferenceMillis) {
        this.angle = angle;
        this.reviewHint = reviewHint;
        this.reviewReason = reviewReason;
        this.inferenceMillis = inferenceMillis;
    }

    public static WitnessStateEstimate experimental(float pointAngle, double inferenceMillis) {
        if (!Float.isFinite(pointAngle) || pointAngle < 0f || pointAngle > 90f
                || !Double.isFinite(inferenceMillis) || inferenceMillis < 0.0) {
            throw new IllegalArgumentException("experimental estimate values are invalid");
        }
        AngleInterval interval = new AngleInterval(
                pointAngle,
                Math.max(0f, pointAngle - EXPERIMENTAL_ERROR_DEGREES),
                Math.min(90f, pointAngle + EXPERIMENTAL_ERROR_DEGREES));
        GeometryDecision routing = AntiLooseGeometry.triagePointAngle(
                pointAngle, ProvisionalTriageThresholds.defaults());
        return new WitnessStateEstimate(
                interval,
                routing.getReviewHint(),
                routing.getReason(),
                inferenceMillis);
    }

    public AngleInterval getAngle() { return angle; }
    public WitnessReviewHint getReviewHint() { return reviewHint; }
    public String getReviewReason() { return reviewReason; }
    public double getInferenceMillis() { return inferenceMillis; }
}
