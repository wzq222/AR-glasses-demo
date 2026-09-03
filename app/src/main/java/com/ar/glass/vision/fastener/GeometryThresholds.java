package com.ar.glass.vision.fastener;

public final class GeometryThresholds {
    private final float maximumAngleDegrees;
    private final float maximumGapRatio;
    private final float maximumResidualRatio;
    private final boolean calibrated;

    public GeometryThresholds(float maximumAngleDegrees, float maximumGapRatio, float maximumResidualRatio) {
        this(maximumAngleDegrees, maximumGapRatio, maximumResidualRatio, true);
    }

    private GeometryThresholds(
            float maximumAngleDegrees,
            float maximumGapRatio,
            float maximumResidualRatio,
            boolean calibrated) {
        if (calibrated && (maximumAngleDegrees < 0f || maximumAngleDegrees > 90f
                || maximumGapRatio < 0f || maximumResidualRatio < 0f)) {
            throw new IllegalArgumentException("Geometry thresholds are outside valid ranges");
        }
        this.maximumAngleDegrees = maximumAngleDegrees;
        this.maximumGapRatio = maximumGapRatio;
        this.maximumResidualRatio = maximumResidualRatio;
        this.calibrated = calibrated;
    }

    public static GeometryThresholds uncalibrated() {
        return new GeometryThresholds(Float.NaN, Float.NaN, Float.NaN, false);
    }

    public float getMaximumAngleDegrees() { return maximumAngleDegrees; }
    public float getMaximumGapRatio() { return maximumGapRatio; }
    public float getMaximumResidualRatio() { return maximumResidualRatio; }
    public boolean isCalibrated() { return calibrated; }
}
