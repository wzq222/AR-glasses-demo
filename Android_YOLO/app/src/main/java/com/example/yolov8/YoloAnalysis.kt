package com.example.yolov8

import android.graphics.Bitmap
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.example.yolov8.core.log.AppLogger
import com.example.yolov8.core.model.DetectorEngine
import java.util.concurrent.atomic.AtomicLong

/**
 * 图像分析器：把相机帧送入检测引擎，回调结果与耗时。
 * 每帧 bitmap 使用后立即回收；ImageProxy 统一在 finally 中关闭。
 */
class YoloAnalysis(
    private val engineProvider: () -> DetectorEngine?,
    private val onResult: (List<BoundingBox>, Long, Int, Int) -> Unit
) : ImageAnalysis.Analyzer {

    private val frameCount = AtomicLong(0)
    private val windowStart = AtomicLong(System.currentTimeMillis())
    private val windowFrames = AtomicLong(0)

    override fun analyze(image: ImageProxy) {
        try {
            val engine = engineProvider() ?: return
            val t0 = System.currentTimeMillis()
            val bitmap = image.toBitmap()
            val frameW = bitmap.width
            val frameH = bitmap.height
            val boxes = try {
                engine.detect(bitmap)
            } finally {
                bitmap.recycle()
            }
            val elapsed = System.currentTimeMillis() - t0

            // FPS 统计（每 15 帧输出一次，避免刷屏）
            val frames = windowFrames.incrementAndGet()
            if (frames % 15 == 0L) {
                val now = System.currentTimeMillis()
                val fps = frames * 1000f / (now - windowStart.getAndSet(now)).coerceAtLeast(1)
                windowFrames.set(0)
                AppLogger.d(
                    "Perf",
                    "frames=${frameCount.addAndGet(frames)} fps=%.1f infer=${elapsed}ms boxes=${boxes.size}"
                        .format(fps)
                )
            }
            onResult(boxes, elapsed, frameW, frameH)
        } catch (e: Exception) {
            AppLogger.w("Analysis", "单帧处理失败: ${e.message}")
        } finally {
            try { image.close() } catch (_: Exception) {}
        }
    }
}