package com.ar.glass.vision.realtime;

public final class SquareRoi {
    private static final float EXPANSION = 1.5f;
    private static final int MINIMUM_SIDE = 16;

    private final int left;
    private final int top;
    private final int side;

    private SquareRoi(int left, int top, int side) {
        this.left = left;
        this.top = top;
        this.side = side;
    }

    public static SquareRoi fromDetection(
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
        float centerX = (detection.getLeft() + detection.getRight()) / 2f;
        float centerY = (detection.getTop() + detection.getBottom()) / 2f;
        if (!Float.isFinite(centerX) || !Float.isFinite(centerY)) {
            throw new IllegalArgumentException("detection center is invalid");
        }
        int maximumSide = Math.min(imageWidth, imageHeight);
        int side = Math.round(Math.min(
                Math.max(Math.max(width, height) * EXPANSION, MINIMUM_SIDE),
                maximumSide));
        int left = clamp(Math.round(centerX - side / 2f), 0, imageWidth - side);
        int top = clamp(Math.round(centerY - side / 2f), 0, imageHeight - side);
        return new SquareRoi(left, top, side);
    }

    public int getLeft() { return left; }
    public int getTop() { return top; }
    public int getRight() { return left + side; }
    public int getBottom() { return top + side; }
    public int getSide() { return side; }

    private static int clamp(int value, int minimum, int maximum) {
        return Math.min(Math.max(value, minimum), maximum);
    }
}
