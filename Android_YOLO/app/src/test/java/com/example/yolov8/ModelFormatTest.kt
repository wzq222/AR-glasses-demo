package com.example.yolov8

import com.example.yolov8.core.model.ModelFormat
import com.example.yolov8.core.model.ModelFormatDetector
import com.example.yolov8.core.model.OnnxNamesParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

class ModelFormatTest {

    private val zipHead = byteArrayOf(0x50, 0x4B, 0x03, 0x04)

    @Test
    fun `onnx 扩展名且非空头部`() {
        val f = ModelFormatDetector.detect("model.onnx", byteArrayOf(0x08, 0x06), emptyList())
        assertEquals(ModelFormat.ONNX, f)
    }

    @Test
    fun `ultralytics pickle checkpoint 识别`() {
        // 形如 screw_detect_v2_best.pt 的真实结构
        val f = ModelFormatDetector.detect(
            "screw_detect_v2_best.pt", zipHead,
            listOf("best/data.pkl", "best/byteorder", "best/data/0", "best/data/1")
        )
        assertEquals(ModelFormat.PICKLE_CHECKPOINT, f)
    }

    @Test
    fun `torchscript 识别 含code目录`() {
        val f = ModelFormatDetector.detect(
            "model_ts.pt", zipHead,
            listOf("model/code/data.pkl", "model/constants.pkl", "model/data/0")
        )
        assertEquals(ModelFormat.TORCHSCRIPT, f)
    }

    @Test
    fun `torchscript 识别 含bytecode`() {
        val f = ModelFormatDetector.detect(
            "model_ts.pt", zipHead,
            listOf("model/bytecode/data.pkl", "model/data/0")
        )
        assertEquals(ModelFormat.TORCHSCRIPT, f)
    }

    @Test
    fun `非zip未知格式`() {
        val f = ModelFormatDetector.detect("model.bin", byteArrayOf(0x00, 0x01, 0x02, 0x03), emptyList())
        assertEquals(ModelFormat.UNKNOWN, f)
    }

    @Test
    fun `真实zip字节流检测`() {
        val bytes = zipOf("best/data.pkl" to "pickle-data".toByteArray())
        val f = ModelFormatDetector.detect("a.pt", bytes, entryNames(bytes))
        assertEquals(ModelFormat.PICKLE_CHECKPOINT, f)
    }

    @Test
    fun `names 解析 标准格式`() {
        val names = OnnxNamesParser.parse("{0: 'screw', 1: 'nut'}")
        assertEquals(listOf("screw", "nut"), names)
    }

    @Test
    fun `names 解析 无引号格式`() {
        val names = OnnxNamesParser.parse("{0: screw}")
        assertEquals(listOf("screw"), names)
    }

    @Test
    fun `names 解析 缺失索引补齐`() {
        val names = OnnxNamesParser.parse("{0: 'a', 2: 'c'}")
        assertEquals(3, names!!.size)
        assertEquals("a", names[0])
        assertEquals("class_1", names[1])
        assertEquals("c", names[2])
    }

    @Test
    fun `names 解析 空值`() {
        assertNull(OnnxNamesParser.parse(null))
        assertNull(OnnxNamesParser.parse(""))
    }

    // ---------- helpers ----------

    private fun zipOf(vararg entries: Pair<String, ByteArray>): ByteArray {
        val bos = ByteArrayOutputStream()
        ZipOutputStream(bos).use { zos ->
            for ((name, data) in entries) {
                zos.putNextEntry(ZipEntry(name))
                zos.write(data)
                zos.closeEntry()
            }
        }
        return bos.toByteArray()
    }

    private fun entryNames(zip: ByteArray): List<String> {
        val list = ArrayList<String>()
        java.util.zip.ZipInputStream(zip.inputStream()).use { zis ->
            while (true) {
                val e = zis.nextEntry ?: break
                list.add(e.name)
                zis.closeEntry()
            }
        }
        assertTrue(list.isNotEmpty())
        return list
    }
}