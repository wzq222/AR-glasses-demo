package com.ar.glass.vision.fastener;

public final class GeometryDecision {
    private final FastenerState state;
    private final String reason;
    private final float angleDegrees;
    private final float gapRatio;
    private final float residualRatio;

    public GeometryDecision(
            FastenerState state,
            String reason,
            float angleDegrees,
            float gapRatio,
            float residualRatio) {
        this.state = state;
        this.reason = reason;
        this.angleDegrees = angleDegrees;
        this.gapRatio = gapRatio;
        this.residualRatio = residualRatio;
    }

    public FastenerState getState() { return state; }
    public String getReason() { return reason; }
    public float getAngleDegrees() { return angleDegrees; }
    public float getGapRatio() { return gapRatio; }
    public float getResidualRatio() { return residualRatio; }
}
