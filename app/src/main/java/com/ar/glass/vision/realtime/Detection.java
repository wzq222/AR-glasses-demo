package com.ar.glass.vision.realtime;

public final class Detection {
    private final float left;
    private final float top;
    private final float right;
    private final float bottom;
    private final float confidence;
    private final int classId;

    Detection(
            float left,
            float top,
            float right,
            float bottom,
            float confidence,
            int classId) {
        this.left = left;
        this.top = top;
        this.right = right;
        this.bottom = bottom;
        this.confidence = confidence;
        this.classId = classId;
    }

    public float getLeft() { return left; }
    public float getTop() { return top; }
    public float getRight() { return right; }
    public float getBottom() { return bottom; }
    public float getConfidence() { return confidence; }
    public int getClassId() { return classId; }
}
