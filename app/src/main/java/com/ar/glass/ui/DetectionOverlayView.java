package com.ar.glass.ui;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Looper;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

import com.ar.glass.vision.realtime.Detection;
import com.ar.glass.vision.realtime.DetectionHitTester;
import com.ar.glass.vision.realtime.PreviewCoordinateMapper;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

public final class DetectionOverlayView extends View {
    private static final int ORANGE = Color.rgb(255, 152, 0);

    private final Paint boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint labelPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint labelBackgroundPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private volatile OverlayResult result = OverlayResult.empty();
    private OnDetectionTapListener detectionTapListener;

    public interface OnDetectionTapListener {
        void onDetectionTapped(Detection detection);
    }

    public DetectionOverlayView(Context context) {
        this(context, null);
    }

    public DetectionOverlayView(Context context, AttributeSet attributes) {
        super(context, attributes);
        float density = getResources().getDisplayMetrics().density;
        float scaledDensity = getResources().getDisplayMetrics().scaledDensity;
        boxPaint.setColor(ORANGE);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(2f * density);
        labelPaint.setColor(Color.BLACK);
        labelPaint.setTextSize(14f * scaledDensity);
        labelBackgroundPaint.setColor(ORANGE);
        labelBackgroundPaint.setStyle(Paint.Style.FILL);
    }

    public void setDetections(List<Detection> detections, int imageWidth, int imageHeight) {
        requireMainThread();
        result = new OverlayResult(detections, imageWidth, imageHeight);
        invalidate();
    }

    public void clearDetections() {
        requireMainThread();
        result = OverlayResult.empty();
        invalidate();
    }

    public void setOnDetectionTapListener(OnDetectionTapListener listener) {
        requireMainThread();
        detectionTapListener = listener;
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        OnDetectionTapListener listener = detectionTapListener;
        OverlayResult snapshot = result;
        if (listener == null
                || snapshot.imageWidth <= 0 || snapshot.imageHeight <= 0
                || getWidth() <= 0 || getHeight() <= 0) {
            return false;
        }
        if (event.getActionMasked() == MotionEvent.ACTION_DOWN) {
            return true;
        }
        if (event.getActionMasked() == MotionEvent.ACTION_UP) {
            Detection selected = DetectionHitTester.smallestContainingFillCenter(
                    snapshot.detections,
                    snapshot.imageWidth,
                    snapshot.imageHeight,
                    getWidth(),
                    getHeight(),
                    event.getX(),
                    event.getY());
            if (selected != null) {
                performClick();
                listener.onDetectionTapped(selected);
            }
            return true;
        }
        return true;
    }

    @Override
    public boolean performClick() {
        super.performClick();
        return true;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        OverlayResult snapshot = result;
        if (snapshot.imageWidth <= 0 || snapshot.imageHeight <= 0
                || getWidth() <= 0 || getHeight() <= 0) {
            return;
        }

        PreviewCoordinateMapper mapper = PreviewCoordinateMapper.fillCenter(
                snapshot.imageWidth, snapshot.imageHeight, getWidth(), getHeight());
        float labelPadding = 4f * getResources().getDisplayMetrics().density;
        Paint.FontMetrics fontMetrics = labelPaint.getFontMetrics();
        float labelHeight = fontMetrics.bottom - fontMetrics.top + labelPadding * 2f;

        for (Detection detection : snapshot.detections) {
            PreviewCoordinateMapper.MappedRect box = mapper.map(
                    detection.getLeft(),
                    detection.getTop(),
                    detection.getRight(),
                    detection.getBottom());
            canvas.drawRect(
                    box.getLeft(), box.getTop(), box.getRight(), box.getBottom(), boxPaint);

            String label = String.format(
                    Locale.CHINA, "候选点 %.2f", detection.getConfidence());
            float labelWidth = labelPaint.measureText(label) + labelPadding * 2f;
            float labelTop = Math.max(0f, box.getTop() - labelHeight);
            canvas.drawRect(
                    box.getLeft(),
                    labelTop,
                    box.getLeft() + labelWidth,
                    labelTop + labelHeight,
                    labelBackgroundPaint);
            canvas.drawText(
                    label,
                    box.getLeft() + labelPadding,
                    labelTop + labelPadding - fontMetrics.top,
                    labelPaint);
        }
    }

    private static void requireMainThread() {
        if (Looper.myLooper() != Looper.getMainLooper()) {
            throw new IllegalStateException("overlay updates must run on the main thread");
        }
    }

    private static final class OverlayResult {
        private final List<Detection> detections;
        private final int imageWidth;
        private final int imageHeight;

        private OverlayResult(List<Detection> detections, int imageWidth, int imageHeight) {
            if (detections == null || imageWidth <= 0 || imageHeight <= 0) {
                throw new IllegalArgumentException("detections and image dimensions are required");
            }
            this.detections = Collections.unmodifiableList(new ArrayList<>(detections));
            this.imageWidth = imageWidth;
            this.imageHeight = imageHeight;
        }

        private OverlayResult() {
            detections = Collections.emptyList();
            imageWidth = 0;
            imageHeight = 0;
        }

        private static OverlayResult empty() {
            return new OverlayResult();
        }
    }
}
