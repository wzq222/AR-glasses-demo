package com.ar.glass.vision.fastener;

public final class AngleInterval {
    private final float pointEstimateDegrees;
    private final float lowerDegrees;
    private final float upperDegrees;

    public AngleInterval(float pointEstimateDegrees, float lowerDegrees, float upperDegrees) {
        if (!isFinite(pointEstimateDegrees)
                || !isFinite(lowerDegrees)
                || !isFinite(upperDegrees)
                || lowerDegrees < 0f
                || lowerDegrees > pointEstimateDegrees
                || pointEstimateDegrees > upperDegrees
                || upperDegrees > 90f) {
            throw new IllegalArgumentException("angle interval is outside valid ranges");
        }
        this.pointEstimateDegrees = pointEstimateDegrees;
        this.lowerDegrees = lowerDegrees;
        this.upperDegrees = upperDegrees;
    }

    public float getPointEstimateDegrees() { return pointEstimateDegrees; }
    public float getLowerDegrees() { return lowerDegrees; }
    public float getUpperDegrees() { return upperDegrees; }

    private static boolean isFinite(float value) {
        return !Float.isNaN(value) && !Float.isInfinite(value);
    }
}
