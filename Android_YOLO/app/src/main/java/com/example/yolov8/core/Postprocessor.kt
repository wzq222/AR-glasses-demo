package com.example.yolov8.core

import kotlin.math.max
import kotlin.math.min

/**
 * letterbox 参数计算（纯函数，便于单元测试）。
 * 保持宽高比缩放至 dst×dst，剩余部分居中填充。
 */
data class LetterboxParams(
    val scale: Float,
    val padX: Float,
    val padY: Float,
    val newW: Int,
    val newH: Int
) {
    companion object {
        fun compute(srcW: Int, srcH: Int, dst: Int): LetterboxParams {
            require(srcW > 0 && srcH > 0 && dst > 0) { "invalid letterbox dims" }
            val scale = min(dst / srcW.toFloat(), dst / srcH.toFloat())
            val newW = max(1, (srcW * scale).toInt())
            val newH = max(1, (srcH * scale).toInt())
            return LetterboxParams(
                scale = scale,
                padX = (dst - newW) / 2f,
                padY = (dst - newH) / 2f,
                newW = newW,
                newH = newH
            )
        }
    }
}

/**
 * 检测框坐标从"分析帧空间"到"预览视图空间"的映射。
 * PreviewView 使用 FILL_CENTER（裁剪填充），需要按缩放裁剪关系换算。
 * 全部使用归一化坐标 [0,1]，纯函数可测。
 */
object CoordinateMapper {

    /**
     * 批量映射一个框：输入帧归一化 (x1,y1,x2,y2)，输出视图归一化四元组。
     * @param fillCenter true=FILL_CENTER 裁剪填充（相机预览）；false=FIT_CENTER 完整显示（虚拟图片）
     */
    fun mapBox(
        x1: Float, y1: Float, x2: Float, y2: Float,
        frameW: Int, frameH: Int, viewW: Int, viewH: Int,
        fillCenter: Boolean = true
    ): FloatArray {
        if (frameW <= 0 || frameH <= 0 || viewW <= 0 || viewH <= 0) {
            return floatArrayOf(x1, y1, x2, y2)
        }

        val sx = viewW.toFloat() / frameW
        val sy = viewH.toFloat() / frameH
        // FILL_CENTER: scale = max（裁剪溢出）；FIT_CENTER: scale = min（留黑边）
        val s = if (fillCenter) max(sx, sy) else min(sx, sy)

        // 帧中心对齐视图中心后，帧在视图坐标系中的偏移
        val offsetX = (viewW - frameW * s) / 2f
        val offsetY = (viewH - frameH * s) / 2f

        fun mx(nx: Float): Float = (nx * frameW * s + offsetX) / viewW
        fun my(ny: Float): Float = (ny * frameH * s + offsetY) / viewH

        return floatArrayOf(
            mx(x1).coerceIn(0f, 1f),
            my(y1).coerceIn(0f, 1f),
            mx(x2).coerceIn(0f, 1f),
            my(y2).coerceIn(0f, 1f)
        )
    }
}

/**
 * 检测结果后处理（纯函数，便于单元测试）。
 * 兼容两种输出布局：
 *  - [1, 4+nc, N]（ultralytics 默认）
 *  - [1, N, 4+nc]（部分导出格式，需转置）
 */
object YoloPostprocessor {

    /** 候选框：x1,y1,x2,y2（帧归一化）+ score + classId */
    data class Candidate(
        val x1: Float, val y1: Float, val x2: Float, val y2: Float,
        val score: Float, val classId: Int
    )

    /**
     * 若输出为 [1, N, C]（N anchors, C 通道），转置为 [C, N] 布局。
     * 判定依据：dim1 > dim2 时认为 dim1 是 anchors 数。
     * @return null 表示无需转置
     */
    fun transposeIfNeeded(data: FloatArray, dim1: Int, dim2: Int): FloatArray? {
        if (dim1 <= 0 || dim2 <= 0 || dim1 * dim2 != data.size) return null
        if (dim1 <= dim2) return null // 已是 [C, N]
        val out = FloatArray(data.size)
        for (n in 0 until dim1) {
            val base = n * dim2
            for (c in 0 until dim2) {
                out[c * dim1 + n] = data[base + c]
            }
        }
        return out
    }

    /**
     * 解码 [C=4+nc, N] 布局输出并按置信度过滤。
     * 坐标为 letterbox 输入空间像素，需用 letterbox 参数还原到帧归一化坐标。
     */
    fun decode(
        data: FloatArray, channels: Int, anchors: Int,
        confThreshold: Float,
        lp: LetterboxParams, frameW: Int, frameH: Int
    ): List<Candidate> {
        if (channels <= 4 || anchors <= 0 || data.size < channels * anchors) return emptyList()
        val numClasses = channels - 4
        val out = ArrayList<Candidate>(64)
        for (i in 0 until anchors) {
            // 取最佳类别分数
            var bestScore = 0f
            var bestCls = 0
            for (c in 0 until numClasses) {
                val sc = data[(4 + c) * anchors + i]
                if (sc > bestScore) { bestScore = sc; bestCls = c }
            }
            if (bestScore < confThreshold) continue

            val cx = data[i]
            val cy = data[anchors + i]
            val w = data[2 * anchors + i]
            val h = data[3 * anchors + i]

            // letterbox 像素 -> 原图像素 -> 归一化
            val px1 = (cx - w / 2f - lp.padX) / lp.scale
            val py1 = (cy - h / 2f - lp.padY) / lp.scale
            val px2 = (cx + w / 2f - lp.padX) / lp.scale
            val py2 = (cy + h / 2f - lp.padY) / lp.scale

            out.add(
                Candidate(
                    x1 = (px1 / frameW).coerceIn(0f, 1f),
                    y1 = (py1 / frameH).coerceIn(0f, 1f),
                    x2 = (px2 / frameW).coerceIn(0f, 1f),
                    y2 = (py2 / frameH).coerceIn(0f, 1f),
                    score = bestScore,
                    classId = bestCls
                )
            )
        }
        return out
    }

    /** IoU（归一化坐标） */
    fun iou(a: Candidate, b: Candidate): Float {
        val ix1 = max(a.x1, b.x1)
        val iy1 = max(a.y1, b.y1)
        val ix2 = min(a.x2, b.x2)
        val iy2 = min(a.y2, b.y2)
        val iw = max(0f, ix2 - ix1)
        val ih = max(0f, iy2 - iy1)
        val inter = iw * ih
        val areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
        val areaB = (b.x2 - b.x1) * (b.y2 - b.y1)
        val union = areaA + areaB - inter
        return if (union <= 0f) 0f else inter / union
    }

    /** 按类别进行 NMS（同类之间抑制） */
    fun nms(candidates: List<Candidate>, iouThreshold: Float): List<Candidate> {
        if (candidates.isEmpty()) return candidates
        val sorted = candidates.sortedByDescending { it.score }
        val keep = ArrayList<Candidate>(sorted.size)
        val removed = BooleanArray(sorted.size)
        for (i in sorted.indices) {
            if (removed[i]) continue
            val a = sorted[i]
            keep.add(a)
            for (j in i + 1 until sorted.size) {
                if (removed[j]) continue
                val b = sorted[j]
                if (a.classId == b.classId && iou(a, b) > iouThreshold) removed[j] = true
            }
        }
        return keep
    }
}
