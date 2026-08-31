package com.ar.glass.vision.fastener;

public final class GeometryDecision {
    private final FastenerState state;
    private final String reason;
    private final float angleDegrees;
    private final float gapRatio;
    private final float residualRatio;
    private final WitnessReviewHint reviewHint;
    private final float angleLowerDegrees;
    private final float angleUpperDegrees;

    public GeometryDecision(
            FastenerState state,
            String reason,
            float angleDegrees,
            float gapRatio,
            float residualRatio) {
        this(
                state,
                reason,
                angleDegrees,
                gapRatio,
                residualRatio,
                WitnessReviewHint.NONE,
                angleDegrees,
                angleDegrees);
    }

    public GeometryDecision(
            FastenerState state,
            String reason,
            float angleDegrees,
            float gapRatio,
            float residualRatio,
            WitnessReviewHint reviewHint,
            float angleLowerDegrees,
            float angleUpperDegrees) {
        this.state = state;
        this.reason = reason;
        this.angleDegrees = angleDegrees;
        this.gapRatio = gapRatio;
        this.residualRatio = residualRatio;
        this.reviewHint = reviewHint;
        this.angleLowerDegrees = angleLowerDegrees;
        this.angleUpperDegrees = angleUpperDegrees;
    }

    public FastenerState getState() { return state; }
    public String getReason() { return reason; }
    public float getAngleDegrees() { return angleDegrees; }
    public float getGapRatio() { return gapRatio; }
    public float getResidualRatio() { return residualRatio; }
    public WitnessReviewHint getReviewHint() { return reviewHint; }
    public float getAngleLowerDegrees() { return angleLowerDegrees; }
    public float getAngleUpperDegrees() { return angleUpperDegrees; }
}
