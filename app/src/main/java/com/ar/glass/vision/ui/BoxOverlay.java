package com.ar.glass.vision.ui;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import com.ar.glass.vision.YoloDetector;
import com.ar.glass.vision.InspectionPresentation;

import java.util.List;
import java.util.Locale;

/**
 * YOLO 检测框实时叠加层（移植自 Android_YOLO 项目的 OverlayView + AnnotationRenderer）。
 *
 * 预览照片使用 fitCenter 完整显示，检测框按同一缩放关系精确映射，
 * 保证框与目标大小一一对应（黑边区域不会出现框）。
 *
 * 渲染优化（相对早期版本）：
 *  - 线宽/字号随框尺寸自适应（小目标细线小字，大目标粗线大字）
 *  - 圆角框 + 淡填充增强辨识
 *  - 标签贴框顶，右侧溢出自动左移，顶部空间不足自动移入框内
 */
public class BoxOverlay extends View {

    private static final int[] BOX_COLORS = {
            0xFFE53935, 0xFF43A047, 0xFF1E88E5, 0xFFFB8C00,
            0xFF8E24AA, 0xFF00ACC1, 0xFFF4511E, 0xFF3949AB
    };

    private volatile List<YoloDetector.Detection> mDetections;
    private volatile int mFrameW;
    private volatile int mFrameH;

    private final Paint mStroke = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint mFill = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint mText = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint mLabelBg = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint mWitness = new Paint(Paint.ANTI_ALIAS_FLAG);

    public BoxOverlay(Context context) {
        super(context);
        init();
    }

    public BoxOverlay(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public BoxOverlay(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    private void init() {
        mStroke.setStyle(Paint.Style.STROKE);
        mText.setColor(Color.WHITE);
        mLabelBg.setStyle(Paint.Style.FILL);
        mWitness.setStyle(Paint.Style.STROKE);
        mWitness.setStrokeCap(Paint.Cap.ROUND);
    }

    /** 提交一轮检测结果（帧为原始照片尺寸，坐标为归一化 [0,1]） */
    public void setResults(List<YoloDetector.Detection> detections, int frameW, int frameH) {
        mDetections = detections;
        mFrameW = frameW;
        mFrameH = frameH;
        postInvalidate();
    }

    public void clear() {
        mDetections = null;
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        List<YoloDetector.Detection> dets = mDetections;
        if (dets == null || dets.isEmpty() || mFrameW <= 0 || mFrameH <= 0) return;

        int viewW = getWidth();
        int viewH = getHeight();
        if (viewW <= 0 || viewH <= 0) return;

        // FIT_CENTER: 照片完整显示（黑边居中），框随同一变换映射
        float sx = viewW / (float) mFrameW;
        float sy = viewH / (float) mFrameH;
        float s = Math.min(sx, sy);
        float offsetX = (viewW - mFrameW * s) / 2f;
        float offsetY = (viewH - mFrameH * s) / 2f;

        for (YoloDetector.Detection d : dets) {
            float x1 = d.x1 * mFrameW * s + offsetX;
            float y1 = d.y1 * mFrameH * s + offsetY;
            float x2 = d.x2 * mFrameW * s + offsetX;
            float y2 = d.y2 * mFrameH * s + offsetY;
            RectF rect = new RectF(x1, y1, x2, y2);
            if (rect.width() < 2 || rect.height() < 2) continue;
            drawBox(canvas, rect, d, viewW);
            drawWitnessSegments(canvas, d, s, offsetX, offsetY);
        }
    }

    private void drawBox(Canvas canvas, RectF rect, YoloDetector.Detection d, int canvasW) {
        int color = stateColor(d);
        float dim = Math.min(rect.width(), rect.height());
        float density = getResources().getDisplayMetrics().density;

        // 自适应线宽 1.2~3.5dp、字号 10~15sp
        float strokeW = clamp(dim / 110f, 1.2f, 3.5f) * density;
        float textSize = clamp(dim / 16f, 10f, 15f) * density;
        float cornerR = clamp(dim * 0.10f, 2f * density, 8f * density);

        mStroke.setColor(color);
        mStroke.setStrokeWidth(strokeW);
        mFill.setColor((color & 0x00FFFFFF) | 0x18000000);
        canvas.drawRoundRect(rect, cornerR, cornerR, mFill);
        canvas.drawRoundRect(rect, cornerR, cornerR, mStroke);

        // 两层结果：第一行定位螺栓，第二行展示防松线几何判断。
        mText.setTextSize(textSize);
        String primary = d.className + " " + String.format(Locale.US, "%.0f%%", d.score * 100f);
        String secondary = d.witnessTriage == null
                ? null
                : InspectionPresentation.stateLabel(d.witnessTriage, d.witnessAngleDegrees);
        float labelW = Math.max(
                mText.measureText(primary),
                secondary == null ? 0f : mText.measureText(secondary));
        float pad = 4f * density;
        float lineHeight = textSize * 1.15f;
        float bgH = lineHeight * (secondary == null ? 1f : 2f) + 2 * pad;

        float labelLeft = rect.left;
        if (labelLeft + labelW + 2 * pad > canvasW) {
            labelLeft = Math.max(0f, canvasW - labelW - 2 * pad);
        }
        float labelTop = rect.top - bgH >= 0 ? rect.top - bgH : rect.top;

        mLabelBg.setColor((color & 0x00FFFFFF) | 0x99000000);
        canvas.drawRoundRect(
                new RectF(labelLeft, labelTop, labelLeft + labelW + 2 * pad, labelTop + bgH),
                cornerR * 0.6f, cornerR * 0.6f, mLabelBg);
        canvas.drawText(primary, labelLeft + pad, labelTop + pad + textSize, mText);
        if (secondary != null) {
            canvas.drawText(
                    secondary,
                    labelLeft + pad,
                    labelTop + pad + textSize + lineHeight,
                    mText);
        }
    }

    private void drawWitnessSegments(
            Canvas canvas,
            YoloDetector.Detection detection,
            float scale,
            float offsetX,
            float offsetY) {
        float[] points = detection.witnessPoints;
        if (points == null || points.length != 8) return;
        int color = stateColor(detection);
        float density = getResources().getDisplayMetrics().density;
        mWitness.setColor(color);
        mWitness.setStrokeWidth(3.2f * density);
        float[] mapped = new float[8];
        for (int index = 0; index < 4; index++) {
            mapped[index * 2] = points[index * 2] * mFrameW * scale + offsetX;
            mapped[index * 2 + 1] = points[index * 2 + 1] * mFrameH * scale + offsetY;
        }
        canvas.drawLine(mapped[0], mapped[1], mapped[2], mapped[3], mWitness);
        canvas.drawLine(mapped[4], mapped[5], mapped[6], mapped[7], mWitness);
        mWitness.setStyle(Paint.Style.FILL);
        for (int index = 0; index < 4; index++) {
            canvas.drawCircle(mapped[index * 2], mapped[index * 2 + 1], 3.5f * density, mWitness);
        }
        mWitness.setStyle(Paint.Style.STROKE);
    }

    private static int stateColor(YoloDetector.Detection detection) {
        if ("LIKELY_ALIGNED".equals(detection.witnessTriage)) return 0xFF2E7D32;
        if ("POSSIBLE_DISPLACED".equals(detection.witnessTriage)) return 0xFFF9A825;
        if ("HIGH_SUSPICION".equals(detection.witnessTriage)) return 0xFFC62828;
        if ("INSUFFICIENT".equals(detection.witnessTriage)) return 0xFF607D8B;
        return BOX_COLORS[Math.abs(detection.classId) % BOX_COLORS.length];
    }

    private static float clamp(float v, float min, float max) {
        return v < min ? min : Math.min(v, max);
    }
}
