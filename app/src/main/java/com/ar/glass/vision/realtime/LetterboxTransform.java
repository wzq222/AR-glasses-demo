package com.ar.glass.vision.realtime;

public final class LetterboxTransform {
    private final int originalWidth;
    private final int originalHeight;
    private final int targetWidth;
    private final int targetHeight;
    private final int resizedWidth;
    private final int resizedHeight;
    private final int padLeft;
    private final int padTop;
    private final int padRight;
    private final int padBottom;
    private final float scale;

    private LetterboxTransform(
            int originalWidth,
            int originalHeight,
            int targetWidth,
            int targetHeight,
            int resizedWidth,
            int resizedHeight,
            int padLeft,
            int padTop,
            int padRight,
            int padBottom,
            float scale) {
        this.originalWidth = originalWidth;
        this.originalHeight = originalHeight;
        this.targetWidth = targetWidth;
        this.targetHeight = targetHeight;
        this.resizedWidth = resizedWidth;
        this.resizedHeight = resizedHeight;
        this.padLeft = padLeft;
        this.padTop = padTop;
        this.padRight = padRight;
        this.padBottom = padBottom;
        this.scale = scale;
    }

    public static LetterboxTransform forSquare(
            int originalWidth, int originalHeight, int targetSize) {
        if (originalWidth <= 0 || originalHeight <= 0 || targetSize <= 0) {
            throw new IllegalArgumentException("image dimensions must be positive");
        }

        float scale = Math.min(
                (float) targetSize / originalWidth,
                (float) targetSize / originalHeight);
        int resizedWidth = Math.round(originalWidth * scale);
        int resizedHeight = Math.round(originalHeight * scale);
        int horizontalPadding = targetSize - resizedWidth;
        int verticalPadding = targetSize - resizedHeight;
        int padLeft = horizontalPadding / 2;
        int padTop = verticalPadding / 2;

        return new LetterboxTransform(
                originalWidth,
                originalHeight,
                targetSize,
                targetSize,
                resizedWidth,
                resizedHeight,
                padLeft,
                padTop,
                horizontalPadding - padLeft,
                verticalPadding - padTop,
                scale);
    }

    public int getOriginalWidth() { return originalWidth; }
    public int getOriginalHeight() { return originalHeight; }
    public int getTargetWidth() { return targetWidth; }
    public int getTargetHeight() { return targetHeight; }
    public int getResizedWidth() { return resizedWidth; }
    public int getResizedHeight() { return resizedHeight; }
    public int getPadLeft() { return padLeft; }
    public int getPadTop() { return padTop; }
    public int getPadRight() { return padRight; }
    public int getPadBottom() { return padBottom; }
    public float getPadX() { return padLeft; }
    public float getPadY() { return padTop; }
    public float getScale() { return scale; }
}
