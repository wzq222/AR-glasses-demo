package com.ar.glass.vision.realtime;

/** Crop coordinates matching the witness-model training _crop_box contract. */
public final class WitnessRoi {
    private static final float EXPANSION = 1.5f;
    private static final float MINIMUM_SIDE = 16f;

    private final int left;
    private final int top;
    private final int right;
    private final int bottom;

    private WitnessRoi(int left, int top, int right, int bottom) {
        this.left = left;
        this.top = top;
        this.right = right;
        this.bottom = bottom;
    }

    public static WitnessRoi fromDetection(
            Detection detection, int imageWidth, int imageHeight) {
        if (detection == null || imageWidth <= 0 || imageHeight <= 0) {
            throw new IllegalArgumentException("detection and image dimensions are required");
        }
        float width = detection.getRight() - detection.getLeft();
        float height = detection.getBottom() - detection.getTop();
        if (!Float.isFinite(width) || !Float.isFinite(height) || width <= 0f || height <= 0f) {
            throw new IllegalArgumentException("detection bounds are invalid");
        }
        if (detection.getRight() <= 0f || detection.getBottom() <= 0f
                || detection.getLeft() >= imageWidth || detection.getTop() >= imageHeight) {
            throw new IllegalArgumentException("detection does not intersect the image");
        }
        float centerX = (detection.getLeft() + detection.getRight()) * 0.5f;
        float centerY = (detection.getTop() + detection.getBottom()) * 0.5f;
        float side = Math.max(Math.max(width, height) * EXPANSION, MINIMUM_SIDE);
        if (!Float.isFinite(centerX) || !Float.isFinite(centerY) || !Float.isFinite(side)) {
            throw new IllegalArgumentException("detection crop is nonfinite");
        }

        int left = pillowRound(Math.max(0f, centerX - side * 0.5f));
        int top = pillowRound(Math.max(0f, centerY - side * 0.5f));
        int right = pillowRound(Math.min((float) imageWidth, centerX + side * 0.5f));
        int bottom = pillowRound(Math.min((float) imageHeight, centerY + side * 0.5f));
        if (right <= left || bottom <= top) {
            throw new IllegalArgumentException("witness crop is empty after frame clamp");
        }
        return new WitnessRoi(left, top, right, bottom);
    }

    public int getLeft() { return left; }
    public int getTop() { return top; }
    public int getRight() { return right; }
    public int getBottom() { return bottom; }
    public int getWidth() { return right - left; }
    public int getHeight() { return bottom - top; }

    private static int pillowRound(float value) {
        return (int) Math.rint(value);
    }
}
