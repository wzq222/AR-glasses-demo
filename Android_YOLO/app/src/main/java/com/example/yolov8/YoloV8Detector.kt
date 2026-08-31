package com.example.yolov8

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import com.example.yolov8.core.CoordinateMapper
import com.example.yolov8.core.log.AppLogger
import java.io.File
import java.util.Locale

/**
 * 检测结果标注渲染：
 *  - [drawOnCanvas] 实时绘制（OverlayView 与虚拟相机预览共用）
 *  - [annotateBitmap] 生成静态标注图（测试模式保存输出用）
 *
 * 坐标系：boxes 为帧归一化坐标；显示时按 FILL_CENTER 裁剪关系映射到视图。
 */
object AnnotationRenderer {

    private val BOX_COLORS = intArrayOf(
        0xFF3E3E.toInt(), 0xFF3ECF3E.toInt(), 0xFF3E8BFF.toInt(),
        0xFFFFC93E.toInt(), 0xFFC33EFF.toInt(), 0xFF3EFFE0.toInt()
    )

    private fun colorFor(classId: Int): Int =
        if (classId < 0) BOX_COLORS[0] else BOX_COLORS[classId.mod(BOX_COLORS.size)]

    fun boxPaint(density: Float, color: Int): Paint = Paint().apply {
        this.color = color
        style = Paint.Style.STROKE
        strokeWidth = 2.5f * density
        isAntiAlias = true
    }

    fun textPaint(density: Float): Paint = Paint().apply {
        color = Color.WHITE
        textSize = 13f * density
        isAntiAlias = true
    }

    fun textBgPaint(density: Float, color: Int): Paint = Paint().apply {
        this.color = color
        isAntiAlias = true
    }

    /**
     * 在视图 canvas 上绘制整组框（帧归一化 → 视图坐标映射在内部完成）。
     * @param fillCenter true=FILL_CENTER（相机预览裁剪填充）；false=FIT_CENTER（虚拟图片完整显示）
     */
    fun drawOnCanvas(
        canvas: Canvas,
        boxes: List<BoundingBox>,
        frameW: Int,
        frameH: Int,
        viewW: Int,
        viewH: Int,
        density: Float,
        fillCenter: Boolean = true
    ) {
        if (boxes.isEmpty() || viewW <= 0 || viewH <= 0) return

        for (b in boxes) {
            val m = CoordinateMapper.mapBox(
                b.x1, b.y1, b.x2, b.y2, frameW, frameH, viewW, viewH, fillCenter
            )
            val rect = RectF(m[0] * viewW, m[1] * viewH, m[2] * viewW, m[3] * viewH)
            if (rect.width() < 2 || rect.height() < 2) continue
            drawBox(canvas, rect, b, canvas.width.toFloat(), density)
        }
    }

    /**
     * 生成静态标注图：在原图上叠加框（无裁剪，直接按 bitmap 尺寸映射）。
     * @return 标注结果（可能是新 bitmap，调用方负责 recycle 返回值；输入不变）
     */
    fun annotateBitmap(src: Bitmap, boxes: List<BoundingBox>, density: Float): Bitmap {
        val out = src.copy(Bitmap.Config.ARGB_8888, true) ?: return src
        val canvas = Canvas(out)
        for (b in boxes) {
            val rect = RectF(
                b.x1 * out.width, b.y1 * out.height,
                b.x2 * out.width, b.y2 * out.height
            )
            if (rect.width() < 2 || rect.height() < 2) continue
            drawBox(canvas, rect, b, out.width.toFloat(), density)
        }
        return out
    }

    /**
     * 单框渲染（两个绘制路径共用）：
     * 线宽/字号随框尺寸自适应——小目标细线小字，大目标粗线大字，保证框与目标视觉大小对应；
     * 圆角框 + 淡填充提高可见性；标签贴框顶，右侧溢出时自动左移。
     */
    private fun drawBox(
        canvas: Canvas,
        rect: RectF,
        box: BoundingBox,
        canvasW: Float,
        density: Float
    ) {
        val color = colorFor(box.classId)
        val dim = minOf(rect.width(), rect.height())

        // 自适应线宽 1.2~3.5dp、字号 10~15sp
        val strokeW = (dim / 110f).coerceIn(1.2f, 3.5f) * density
        val textSize = (dim / 16f).coerceIn(10f, 15f) * density
        val cornerR = (dim * 0.10f).coerceIn(2f * density, 8f * density)

        val stroke = boxPaint(density, color).apply { strokeWidth = strokeW }
        // 淡填充增强框内辨识（不遮挡目标）
        val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            this.color = (color and 0x00FFFFFF) or 0x18000000
            style = Paint.Style.FILL
        }
        canvas.drawRoundRect(rect, cornerR, cornerR, fill)
        canvas.drawRoundRect(rect, cornerR, cornerR, stroke)

        // 标签
        val text = textPaint(density).apply { this.textSize = textSize }
        val label = "${box.className} ${String.format(Locale.US, "%.2f", box.score)}"
        val labelW = text.measureText(label)
        val pad = 4f * density
        val bgH = textSize + 2 * pad

        // 标签左缘：默认贴框左；右溢出时贴画布右缘
        var labelLeft = rect.left
        if (labelLeft + labelW + 2 * pad > canvasW) {
            labelLeft = (canvasW - labelW - 2 * pad).coerceAtLeast(0f)
        }
        // 标签在框顶上方，顶部空间不足时放框内顶部
        val labelTop = if (rect.top - bgH >= 0) rect.top - bgH else rect.top

        val bg = textBgPaint(density, (color and 0x00FFFFFF) or 0x99000000.toInt())
        canvas.drawRoundRect(
            RectF(labelLeft, labelTop, labelLeft + labelW + 2 * pad, labelTop + bgH),
            cornerR * 0.6f, cornerR * 0.6f, bg
        )
        canvas.drawText(label, labelLeft + pad, labelTop + pad + textSize * 0.86f, text)
    }

    /** 保存标注图到公共 Pictures/YOLOTest（API29+ 无需权限；API≤28 需 WRITE 权限） */
    fun saveAnnotated(context: Context, bitmap: Bitmap, baseName: String): String? {
        return try {
            val resolver = context.contentResolver
            if (android.os.Build.VERSION.SDK_INT >= 29) {
                val values = android.content.ContentValues().apply {
                    put(android.provider.MediaStore.Images.Media.DISPLAY_NAME,
                        "$baseName.jpg")
                    put(android.provider.MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                    put(android.provider.MediaStore.Images.Media.RELATIVE_PATH,
                        "Pictures/YOLOTest")
                }
                val uri = resolver.insert(
                    android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values
                ) ?: return null
                resolver.openOutputStream(uri)?.use { os ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 92, os)
                }
                uri.toString()
            } else {
                @Suppress("DEPRECATION")
                val dir = android.os.Environment.getExternalStoragePublicDirectory(
                    android.os.Environment.DIRECTORY_PICTURES
                )
                val sub = File(dir, "YOLOTest")
                if (!sub.exists()) sub.mkdirs()
                val f = File(sub, "$baseName.jpg")
                f.outputStream().use { os ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 92, os)
                }
                f.absolutePath
            }
        } catch (e: Exception) {
            AppLogger.e("Save", "保存标注图失败", e)
            null
        }
    }
}