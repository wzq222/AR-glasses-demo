package com.ar.glass.vision.fastener;

public final class VisionPoint {
    private final float x;
    private final float y;
    private final float confidence;

    public VisionPoint(float x, float y, float confidence) {
        this.x = x;
        this.y = y;
        this.confidence = confidence;
    }

    public float getX() { return x; }
    public float getY() { return y; }
    public float getConfidence() { return confidence; }
}
