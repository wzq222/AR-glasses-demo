package com.example.yolov8

import android.content.Context
import android.graphics.Canvas
import android.util.AttributeSet
import android.view.View
import java.util.concurrent.atomic.AtomicReference

/**
 * 检测框实时绘制层。
 * 通过 [setResults] 提交结果与帧尺寸，绘制时按 FILL_CENTER 裁剪关系映射到视图坐标。
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private data class Frame(
        val boxes: List<BoundingBox>,
        val frameW: Int,
        val frameH: Int
    )

    private val frame = AtomicReference<Frame?>(null)

    /** 帧到视图的缩放模式：true=FILL_CENTER（相机），false=FIT_CENTER（虚拟图片） */
    @Volatile
    private var fillCenter = true

    /** 按显示模式切换映射（模式切换时调用，立即重绘） */
    fun setFillCenter(fill: Boolean) {
        fillCenter = fill
        postInvalidate()
    }

    fun setResults(boxes: List<BoundingBox>, frameW: Int, frameH: Int) {
        frame.set(Frame(boxes, frameW, frameH))
        postInvalidate()
    }

    fun clear() {
        frame.set(null)
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val f = frame.get() ?: return
        AnnotationRenderer.drawOnCanvas(
            canvas, f.boxes, f.frameW, f.frameH, width, height,
            resources.displayMetrics.density, fillCenter
        )
    }
}