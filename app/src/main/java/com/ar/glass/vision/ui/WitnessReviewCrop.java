package com.ar.glass.vision.ui;

/**
 * Source-image crop geometry for one human-reviewed witness-line target.
 * This is intentionally independent from the 1.5x crop used by the state model.
 */
public final class WitnessReviewCrop {
    public static final float EXPANSION = 2.25f;
    public static final float MIN_REVIEWABLE_SIDE = 32f;

    private final float requestedLeft;
    private final float requestedTop;
    private final float side;
    private final float roiLeft;
    private final float roiTop;
    private final float roiRight;
    private final float roiBottom;
    private final boolean requiresCloserCapture;

    private WitnessReviewCrop(
            float requestedLeft,
            float requestedTop,
            float side,
            float roiLeft,
            float roiTop,
            float roiRight,
            float roiBottom,
            boolean requiresCloserCapture) {
        this.requestedLeft = requestedLeft;
        this.requestedTop = requestedTop;
        this.side = side;
        this.roiLeft = roiLeft;
        this.roiTop = roiTop;
        this.roiRight = roiRight;
        this.roiBottom = roiBottom;
        this.requiresCloserCapture = requiresCloserCapture;
    }

    public static WitnessReviewCrop fromNormalized(
            float left,
            float top,
            float right,
            float bottom,
            int imageWidth,
            int imageHeight) {
        if (imageWidth <= 0 || imageHeight <= 0
                || !Float.isFinite(left) || !Float.isFinite(top)
                || !Float.isFinite(right) || !Float.isFinite(bottom)
                || right <= left || bottom <= top) {
            throw new IllegalArgumentException("valid normalized ROI and image size required");
        }
        float x1 = left * imageWidth;
        float y1 = top * imageHeight;
        float x2 = right * imageWidth;
        float y2 = bottom * imageHeight;
        float targetSide = Math.max(x2 - x1, y2 - y1);
        float side = Math.max(1f, targetSide * EXPANSION);
        float cropLeft = (x1 + x2 - side) * 0.5f;
        float cropTop = (y1 + y2 - side) * 0.5f;
        return new WitnessReviewCrop(
                cropLeft,
                cropTop,
                side,
                (x1 - cropLeft) / side,
                (y1 - cropTop) / side,
                (x2 - cropLeft) / side,
                (y2 - cropTop) / side,
                targetSide < MIN_REVIEWABLE_SIDE);
    }

    public float getRequestedLeft() { return requestedLeft; }
    public float getRequestedTop() { return requestedTop; }
    public float getSide() { return side; }
    public float getRoiLeft() { return roiLeft; }
    public float getRoiTop() { return roiTop; }
    public float getRoiRight() { return roiRight; }
    public float getRoiBottom() { return roiBottom; }
    public boolean requiresCloserCapture() { return requiresCloserCapture; }
}
