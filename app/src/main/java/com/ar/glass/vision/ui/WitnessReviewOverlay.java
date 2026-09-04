package com.ar.glass.vision.ui;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

/** Draws the source detection ROI over a centered human-review crop. */
public final class WitnessReviewOverlay extends View {
    private final Paint stroke = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint label = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint labelBackground = new Paint(Paint.ANTI_ALIAS_FLAG);
    private WitnessReviewCrop crop;
    private int pointIndex;
    private String triage;

    public WitnessReviewOverlay(Context context) {
        super(context);
        initialize();
    }

    public WitnessReviewOverlay(Context context, AttributeSet attrs) {
        super(context, attrs);
        initialize();
    }

    public WitnessReviewOverlay(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        initialize();
    }

    private void initialize() {
        stroke.setStyle(Paint.Style.STROKE);
        stroke.setStrokeWidth(3f * getResources().getDisplayMetrics().density);
        label.setColor(Color.WHITE);
        label.setTextSize(13f * getResources().getDisplayMetrics().scaledDensity);
        labelBackground.setStyle(Paint.Style.FILL);
    }

    public void setReview(WitnessReviewCrop crop, int pointIndex, String triage) {
        this.crop = crop;
        this.pointIndex = pointIndex;
        this.triage = triage;
        invalidate();
    }

    public void clear() {
        crop = null;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (crop == null || getWidth() <= 0 || getHeight() <= 0) return;
        float side = Math.min(getWidth(), getHeight());
        float offsetX = (getWidth() - side) * 0.5f;
        float offsetY = (getHeight() - side) * 0.5f;
        RectF roi = new RectF(
                offsetX + crop.getRoiLeft() * side,
                offsetY + crop.getRoiTop() * side,
                offsetX + crop.getRoiRight() * side,
                offsetY + crop.getRoiBottom() * side);
        int color = triageColor(triage);
        stroke.setColor(color);
        canvas.drawRect(roi, stroke);

        String text = "ROI " + pointIndex;
        float padding = 4f * getResources().getDisplayMetrics().density;
        float textWidth = label.measureText(text);
        float height = label.getTextSize() + padding * 2f;
        float left = Math.max(offsetX, roi.left);
        float top = roi.top - height >= offsetY ? roi.top - height : roi.top;
        labelBackground.setColor((color & 0x00FFFFFF) | 0xCC000000);
        canvas.drawRect(left, top, left + textWidth + padding * 2f, top + height, labelBackground);
        canvas.drawText(text, left + padding, top + padding + label.getTextSize() * 0.86f, label);
    }

    private static int triageColor(String triage) {
        if ("LIKELY_ALIGNED".equals(triage)) return 0xFF2E7D32;
        if ("POSSIBLE_DISPLACED".equals(triage)) return 0xFFF9A825;
        if ("HIGH_SUSPICION".equals(triage)) return 0xFFC62828;
        return 0xFF00ACC1;
    }
}
