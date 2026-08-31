package com.ar.glass.vision.ui;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import com.ar.glass.vision.YoloDetector;

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
        }
    }

    private void drawBox(Canvas canvas, RectF rect, YoloDetector.Detection d, int canvasW) {
        int color = BOX_COLORS[Math.abs(d.classId) % BOX_COLORS.length];
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

        // 标签：类名 + 置信度
        mText.setTextSize(textSize);
        String label = d.className + " " + String.format(Locale.US, "%.2f", d.score);
        float labelW = mText.measureText(label);
        float pad = 4f * density;
        float bgH = textSize + 2 * pad;

        float labelLeft = rect.left;
        if (labelLeft + labelW + 2 * pad > canvasW) {
            labelLeft = Math.max(0f, canvasW - labelW - 2 * pad);
        }
        float labelTop = rect.top - bgH >= 0 ? rect.top - bgH : rect.top;

        mLabelBg.setColor((color & 0x00FFFFFF) | 0x99000000);
        canvas.drawRoundRect(
                new RectF(labelLeft, labelTop, labelLeft + labelW + 2 * pad, labelTop + bgH),
                cornerR * 0.6f, cornerR * 0.6f, mLabelBg);
        canvas.drawText(label, labelLeft + pad, labelTop + pad + textSize * 0.86f, mText);
    }

    private static float clamp(float v, float min, float max) {
        return v < min ? min : Math.min(v, max);
    }
}
