package com.ar.glass.vision.realtime;

public final class PreviewCoordinateMapper {
    private final float scale;
    private final float offsetX;
    private final float offsetY;

    private PreviewCoordinateMapper(float scale, float offsetX, float offsetY) {
        this.scale = scale;
        this.offsetX = offsetX;
        this.offsetY = offsetY;
    }

    public static PreviewCoordinateMapper fillCenter(
            int imageWidth,
            int imageHeight,
            int previewWidth,
            int previewHeight) {
        if (imageWidth <= 0 || imageHeight <= 0
                || previewWidth <= 0 || previewHeight <= 0) {
            throw new IllegalArgumentException("image and preview dimensions must be positive");
        }
        float scale = Math.max(
                previewWidth / (float) imageWidth,
                previewHeight / (float) imageHeight);
        float offsetX = (previewWidth - imageWidth * scale) / 2f;
        float offsetY = (previewHeight - imageHeight * scale) / 2f;
        return new PreviewCoordinateMapper(scale, offsetX, offsetY);
    }

    public MappedRect map(float left, float top, float right, float bottom) {
        return new MappedRect(
                left * scale + offsetX,
                top * scale + offsetY,
                right * scale + offsetX,
                bottom * scale + offsetY);
    }

    public static final class MappedRect {
        private final float left;
        private final float top;
        private final float right;
        private final float bottom;

        private MappedRect(float left, float top, float right, float bottom) {
            this.left = left;
            this.top = top;
            this.right = right;
            this.bottom = bottom;
        }

        public float getLeft() { return left; }
        public float getTop() { return top; }
        public float getRight() { return right; }
        public float getBottom() { return bottom; }
    }
}
