package com.example.yolov8

import com.example.yolov8.core.CoordinateMapper
import com.example.yolov8.core.LetterboxParams
import com.example.yolov8.core.YoloPostprocessor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class PostprocessorTest {

    // ---------- LetterboxParams ----------

    @Test
    fun `letterbox 宽图填充垂直`() {
        val lp = LetterboxParams.compute(1920, 1080, 640)
        // 宽图：宽度方向撑满，垂直留白
        assertEquals(640, lp.newW)
        assertEquals(360, lp.newH)
        assertEquals(0f, lp.padX)
        assertEquals((640 - 360) / 2f, lp.padY)
        assertEquals(640f / 1920f, lp.scale, 1e-6f)
    }

    @Test
    fun `letterbox 竖图填充水平`() {
        val lp = LetterboxParams.compute(1080, 1920, 640)
        assertEquals(360, lp.newW)
        assertEquals(640, lp.newH)
        assertEquals((640 - 360) / 2f, lp.padX)
        assertEquals(0f, lp.padY)
    }

    @Test
    fun `letterbox 正方形无填充`() {
        val lp = LetterboxParams.compute(640, 640, 640)
        assertEquals(0f, lp.padX)
        assertEquals(0f, lp.padY)
        assertEquals(1f, lp.scale, 1e-6f)
    }

    // ---------- decode ----------

    /** 构造 [C=4+numClasses, N] 布局：anchors i 处中心 (i*10, i*10) 尺寸 20x20 */
    private fun makeOutput(
        anchors: Int, numClasses: Int,
        activations: Map<Int, Pair<Int, Float>> // anchorIdx -> (classId, score)
    ): FloatArray {
        val c = 4 + numClasses
        val data = FloatArray(c * anchors)
        for (i in 0 until anchors) {
            data[i] = i * 10f // cx
            data[anchors + i] = i * 10f // cy
            data[2 * anchors + i] = 20f // w
            data[3 * anchors + i] = 20f // h
        }
        for ((i, act) in activations) {
            data[(4 + act.first) * anchors + i] = act.second
        }
        return data
    }

    @Test
    fun `decode 过滤低置信度并还原坐标`() {
        // 多类输出：anchor 3 → class 2 score 0.9（通过阈值），anchor 1 → class 0 score 0.2（被过滤）
        val anchors = 10
        val data = makeOutput(anchors, 3, mapOf(3 to (2 to 0.9f), 1 to (0 to 0.2f)))
        // 模拟：帧 640x640 直接输入（scale=1, pad=0）
        val lp = LetterboxParams.compute(640, 640, 640)
        val cands = YoloPostprocessor.decode(data, 7, anchors, 0.5f, lp, 640, 640)

        assertEquals(1, cands.size)
        val c = cands[0]
        assertEquals(2, c.classId)
        assertEquals(0.9f, c.score, 1e-6f)
        // cx=30 cy=30 w=20 h=20 → x1=20,y1=20,x2=40,y2=40 → 归一化 /640
        assertEquals(20f / 640f, c.x1, 1e-4f)
        assertEquals(40f / 640f, c.x2, 1e-4f)
    }

    @Test
    fun `decode 支持转置后的 CN 布局`() {
        // 模拟真实 [N, C] 输出（如 8400x85 的 N>>C 关系）：N=8, C=7（4+3类）
        val anchors = 8
        val c = 7
        val ncRaw = FloatArray(anchors * c)
        for (n in 0 until anchors) {
            ncRaw[n * c + 0] = n * 10f // cx
            ncRaw[n * c + 1] = n * 10f // cy
            ncRaw[n * c + 2] = 20f     // w
            ncRaw[n * c + 3] = 20f     // h
            ncRaw[n * c + 4 + (n % 3)] = 0.8f // 每行激活一个类别
        }
        val transposed = YoloPostprocessor.transposeIfNeeded(ncRaw, anchors, c)
        assertNotNull(transposed)
        val lp = LetterboxParams.compute(640, 640, 640)
        val cands = YoloPostprocessor.decode(transposed!!, c, anchors, 0.5f, lp, 640, 640)
        assertEquals(anchors, cands.size)
        // 三个类别均应出现，且类别与 n%3 对应
        assertEquals(setOf(0, 1, 2), cands.map { it.classId }.toSet())
        // anchor 1（cx=10,w=20）→ class 1，x1 = 0
        val c1 = cands.first { it.classId == 1 }
        assertEquals(0f, c1.x1, 1e-4f)
    }

    @Test
    fun `transposeIfNeeded N大于C时执行转置且结果正确`() {
        // [N=5, C=4]（dim1 > dim2，模拟 8400x85 关系）→ 转置为 [C=4, N=5]
        val src = FloatArray(20) { it.toFloat() }
        val out = YoloPostprocessor.transposeIfNeeded(src, 5, 4)
        assertNotNull(out)
        // src[n*4+c] → out[c*5+n]
        assertEquals(src[0], out!![0], 1e-6f) // n=0,c=0
        assertEquals(src[1], out[5], 1e-6f) // n=0,c=1 → out[1*5+0]
        assertEquals(src[6], out[11], 1e-6f) // n=1,c=2 → out[2*5+1]
    }

    @Test
    fun `transposeIfNeeded 已是 CN 布局时返回 null`() {
        // [C=4, N=5]（dim1 <= dim2，模拟 5x8400 关系）→ 无需转置
        assertNull(YoloPostprocessor.transposeIfNeeded(FloatArray(20), 4, 5))
    }

    // ---------- IoU / NMS ----------

    private fun cand(x1: Float, y1: Float, x2: Float, y2: Float, s: Float, cls: Int = 0) =
        YoloPostprocessor.Candidate(x1, y1, x2, y2, s, cls)

    @Test
    fun `iou 完全重叠为1 不相交为0`() {
        val a = cand(0f, 0f, 0.5f, 0.5f, 0.9f)
        assertEquals(1f, YoloPostprocessor.iou(a, a), 1e-6f)
        val b = cand(0.6f, 0.6f, 1f, 1f, 0.8f)
        assertEquals(0f, YoloPostprocessor.iou(a, b), 1e-6f)
    }

    @Test
    fun `iou 半重叠`() {
        val a = cand(0f, 0f, 0.5f, 0.5f, 0.9f)
        val b = cand(0.25f, 0f, 0.75f, 0.5f, 0.8f)
        // inter = 0.25*0.5 = 0.125, union = 0.25+0.25-0.125 = 0.375
        assertEquals(0.125f / 0.375f, YoloPostprocessor.iou(a, b), 1e-5f)
    }

    @Test
    fun `nms 抑制同类高重叠 保留异类`() {
        val base = cand(0f, 0f, 0.5f, 0.5f, 0.9f, 0)
        val dup = cand(0.01f, 0.01f, 0.51f, 0.51f, 0.85f, 0)
        val other = cand(0.01f, 0.01f, 0.51f, 0.51f, 0.7f, 1) // 同位置不同类
        val keep = YoloPostprocessor.nms(listOf(base, dup, other), 0.45f)
        assertEquals(2, keep.size)
        assertTrue(keep.any { it.classId == 0 && it.score == 0.9f })
        assertTrue(keep.any { it.classId == 1 })
    }

    @Test
    fun `nms 低重叠框都保留`() {
        val a = cand(0f, 0f, 0.3f, 0.3f, 0.9f)
        val b = cand(0.5f, 0.5f, 0.8f, 0.8f, 0.8f)
        assertEquals(2, YoloPostprocessor.nms(listOf(a, b), 0.45f).size)
    }

    // ---------- CoordinateMapper fit/fill 映射 ----------

    @Test
    fun `mapBox fillCenter 宽图水平裁剪 框随比例放大`() {
        // 帧 1920x1080 → 视图 640x640：fill s=max(1/3, 0.5926)=0.5926（纵向决定）
        // 帧显示宽 1137.8 > 640，offsetX = (640-1137.8)/2 = -248.9
        val m = CoordinateMapper.mapBox(0.25f, 0.25f, 0.75f, 0.75f, 1920, 1080, 640, 640, true)
        // x=0.25*1920*0.5926=284.4-248.9=35.6 → /640=0.0556；对称 → x2=0.9444
        assertEquals(0.0556f, m[0], 1e-3f)
        assertEquals(0.9444f, m[2], 1e-3f)
        // y 无裁剪：0.25 直传
        assertEquals(0.25f, m[1], 1e-3f)
        assertEquals(0.75f, m[3], 1e-3f)
    }

    @Test
    fun `mapBox fitCenter 宽图上下留黑边 框完整对应`() {
        // 同样帧 1920x1080 → 视图 640x640：fit s=min=1/3，黑边 (640-640)/2=0 垂直，
        // 水平黑边 (640-1920/3)/2 = (640-640)/2 = 0 … 用竖图验证黑边更直观
        val m = CoordinateMapper.mapBox(0f, 0f, 1f, 1f, 1080, 1920, 640, 640, false)
        // fit s = 640/1920 = 1/3 → 帧显示 360x640，水平黑边 140
        // 全帧框 → 左右各留 140/640 = 0.219 黑边
        assertEquals(140f / 640f, m[0], 1e-3f)
        assertEquals(1f - 140f / 640f, m[2], 1e-3f)
        assertEquals(0f, m[1], 1e-4f)
        assertEquals(1f, m[3], 1e-4f)
    }

    @Test
    fun `mapBox fill 与 fit 对同框给出不同尺寸`() {
        val f = CoordinateMapper.mapBox(0.1f, 0.1f, 0.9f, 0.9f, 1080, 1920, 640, 640, true)
        val t = CoordinateMapper.mapBox(0.1f, 0.1f, 0.9f, 0.9f, 1080, 1920, 640, 640, false)
        // fill（裁剪）框应大于 fit（完整缩小）框
        assertTrue(f[2] - f[0] > t[2] - t[0])
        assertTrue(f[3] - f[1] > t[3] - t[1])
    }

    // ---------- CoordinateMapper ----------

    @Test
    fun `映射 宽高一致时线性直传`() {
        val m = CoordinateMapper.mapBox(0.25f, 0.25f, 0.75f, 0.75f, 640, 640, 320, 320)
        assertEquals(0.25f, m[0], 1e-5f)
        assertEquals(0.25f, m[1], 1e-5f)
        assertEquals(0.75f, m[2], 1e-5f)
        assertEquals(0.75f, m[3], 1e-5f)
    }

    @Test
    fun `映射 视图更宽时垂直裁剪`() {
        // 帧 4:3 (640x480)，视图 16:9 (640x360)
        // scale = max(640/640, 360/480) = max(1, 0.75) = 1 → 水平撑满，垂直裁剪
        // 帧高 480*1=480 映射到 360 视图：offsetY = (360-480)/2 = -60
        val m = CoordinateMapper.mapBox(0f, 0.25f, 1f, 0.75f, 640, 480, 640, 360)
        assertEquals(0f, m[0], 1e-5f)
        // y=0.25*480=120 → 120-60=60 → /360 = 1/6
        assertEquals(60f / 360f, m[1], 1e-4f)
        // y=0.75*480=360 → 360-60=300 → /360 = 5/6
        assertEquals(300f / 360f, m[2] * 0 + (m[3] - 0f), 1e-4f)
        assertTrue(abs(m[3] - 300f / 360f) < 1e-4)
    }

    @Test
    fun `映射 视图更窄时水平裁剪`() {
        // 帧 16:9 (1920x1080)，视图 4:3 (480x360)
        // scale = max(480/1920=0.25, 360/1080=0.3333) = 0.3333 → 垂直撑满，水平裁剪
        val m = CoordinateMapper.mapBox(0.5f, 0f, 0.5f, 1f, 1920, 1080, 480, 360)
        // x=0.5*1920=960 *0.3333=320 → offsetX=(480-1920*0.3333)/2=(480-640)/2=-80 → 320-80=240 → /480=0.5
        assertEquals(0.5f, m[0], 1e-3f)
    }

    @Test
    fun `映射 非法尺寸回退原值`() {
        val m = CoordinateMapper.mapBox(0.1f, 0.2f, 0.3f, 0.4f, 0, 0, 100, 100)
        assertTrue(m.contentEquals(floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f)))
    }
}