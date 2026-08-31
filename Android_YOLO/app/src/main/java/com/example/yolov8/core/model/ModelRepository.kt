package com.example.yolov8.core.model

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import com.example.yolov8.BoundingBox
import com.example.yolov8.core.LetterboxParams
import com.example.yolov8.core.YoloPostprocessor
import com.example.yolov8.core.log.AppLogger
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.InputStream
import java.nio.FloatBuffer
import java.util.Collections
import java.util.regex.Pattern

private const val TAG = "Model"

/** 模型文件格式 */
enum class ModelFormat { ONNX, TORCHSCRIPT, PICKLE_CHECKPOINT, UNKNOWN }

/**
 * 格式检测（纯 JVM 可测）：
 *  - .onnx 扩展名 → ONNX
 *  - zip(PK) 且含 data.pkl：
 *      - 含 code/bytecode/constants.pkl → TorchScript（可被 PyTorch Mobile 直接加载）
 *      - 否则 → Ultralytics pickle checkpoint（Android 无法直接反序列化，需转换）
 */
object ModelFormatDetector {

    private const val ZIP_MAGIC_1 = 0x50 // 'P'
    private const val ZIP_MAGIC_2 = 0x4B // 'K'

    fun detect(fileName: String, head: ByteArray, zipEntryNames: List<String>): ModelFormat {
        val lower = fileName.lowercase()
        if (lower.endsWith(".onnx")) {
            return if (head.isEmpty()) ModelFormat.UNKNOWN else ModelFormat.ONNX
        }
        if (head.size < 4 || head[0].toInt() != ZIP_MAGIC_1 || head[1].toInt() != ZIP_MAGIC_2) {
            return ModelFormat.UNKNOWN
        }
        val hasDataPkl = zipEntryNames.any { it.endsWith("data.pkl") }
        val hasCode = zipEntryNames.any {
            it.endsWith("constants.pkl") || it.contains("/code/") ||
                it.endsWith("/code") || it.contains("/bytecode/")
        }
        return when {
            hasDataPkl && hasCode -> ModelFormat.TORCHSCRIPT
            !hasDataPkl && hasCode -> ModelFormat.TORCHSCRIPT
            hasDataPkl -> ModelFormat.PICKLE_CHECKPOINT
            else -> ModelFormat.UNKNOWN
        }
    }

    fun isPytorchFile(fileName: String): Boolean = fileName.lowercase().endsWith(".pt")
}

/** 模型加载/推理异常（携带用户可读信息） */
sealed class ModelException(message: String, cause: Throwable? = null) : Exception(message, cause) {
    class FileNotFound(path: String) : ModelException("模型文件不存在: $path")
    class EmptyFile(path: String) : ModelException("模型文件为空: $path")
    class UnsupportedFormat(format: ModelFormat, detail: String) :
        ModelException(buildString {
            append("不支持的模型格式: $format。")
            append(detail)
        })

    class InvalidOnnx(detail: String) : ModelException("无效的 ONNX 模型: $detail")
    class Inference(detail: String, cause: Throwable? = null) : ModelException("推理失败: $detail", cause)
}

/** 模型元信息 */
data class ModelInfo(
    val fileName: String,
    val format: ModelFormat,
    val inputSize: Int,
    val classNames: List<String>,
    val accelerator: String,
    val loadMs: Long,
    /** 硬件加速是否实际生效（NNAPI 初始化成功；失败回退 CPU 时为 false） */
    val gpuActive: Boolean = false
)

/**
 * 检测引擎抽象：统一 ONNX 与 TorchScript 推理路径。
 */
interface DetectorEngine : AutoCloseable {
    val info: ModelInfo
    var confThreshold: Float
    var iouThreshold: Float

    /** 输入 bitmap 不会被回收，由调用方管理生命周期 */
    fun detect(bitmap: Bitmap): List<BoundingBox>
}

/** ultralytics ONNX metadata "names" 字段解析（纯 JVM 可测） */
object OnnxNamesParser {
    private val ITEM = Pattern.compile("(\\d+)\\s*:\\s*'?([^,'}\"']+)")

    /** 形如 "{0: 'screw', 1: 'nut'}" 或 "{0: screw, 1: nut}" */
    fun parse(raw: String?): List<String>? {
        if (raw.isNullOrBlank()) return null
        val m = ITEM.matcher(raw)
        val pairs = ArrayList<Pair<Int, String>>()
        while (m.find()) {
            pairs.add(m.group(1)!!.toInt() to m.group(2)!!.trim())
        }
        if (pairs.isEmpty()) return null
        val maxIdx = pairs.maxOf { it.first }
        val out = arrayOfNulls<String>(maxIdx + 1)
        for ((idx, name) in pairs) out[idx] = name
        return out.map { it ?: "class_${pairs.indexOfFirst { false }}" }.let { list ->
            List(out.size) { i -> out[i] ?: "class_$i" }
        }
    }
}

/**
 * ONNX Runtime 引擎（CPU 或 NNAPI 硬件加速）。
 */
class OnnxEngine private constructor(
    private val env: OrtEnvironment,
    session: OrtSession,
    override val info: ModelInfo,
    confThreshold: Float,
    iouThreshold: Float
) : DetectorEngine {

    companion object {
        fun create(
            modelBytes: ByteArray,
            fileName: String,
            fallbackLabels: List<String>,
            useGpu: Boolean,
            confThreshold: Float,
            iouThreshold: Float
        ): OnnxEngine {
            val t0 = System.currentTimeMillis()
            val env = OrtEnvironment.getEnvironment()

            // NNAPI 硬件加速：初始化失败自动回退 CPU，并记录 gpuActive 供 UI 提示
            var gpuActive = false
            val session = try {
                if (useGpu) {
                    try {
                        val gpuOpts = OrtSession.SessionOptions()
                        gpuOpts.setIntraOpNumThreads(4)
                        gpuOpts.addNnapi() // GPU/NPU 加速，兼容性优先
                        val s = env.createSession(modelBytes, gpuOpts)
                        gpuActive = true
                        s
                    } catch (e: Exception) {
                        AppLogger.w(TAG, "NNAPI 初始化失败，回退 CPU: ${e.message}")
                        val cpuOpts = OrtSession.SessionOptions()
                        cpuOpts.setIntraOpNumThreads(4)
                        env.createSession(modelBytes, cpuOpts)
                    }
                } else {
                    val cpuOpts = OrtSession.SessionOptions()
                    cpuOpts.setIntraOpNumThreads(4)
                    env.createSession(modelBytes, cpuOpts)
                }
            } catch (e: Exception) {
                throw ModelException.InvalidOnnx(e.message ?: "createSession failed")
            }

            // 输入尺寸：优先读取固定 shape，符号维度回退 640
            val firstInputInfo = session.inputInfo.values.first().info
            val shape = (firstInputInfo as? ai.onnxruntime.TensorInfo)?.shape
            var inputSize = 640
            if (shape != null && shape.size >= 4 && shape[2] > 0 && shape[3] > 0) {
                inputSize = shape[2].toInt()
            }

            // 类别名：metadata "names" → fallback labels → class_i
            val metaNames = try {
                OnnxNamesParser.parse(session.metadata.customMetadata["names"])
            } catch (_: Exception) { null }
            val names = metaNames ?: fallbackLabels

            val loadMs = System.currentTimeMillis() - t0
            return OnnxEngine(
                env, session,
                ModelInfo(
                    fileName = fileName,
                    format = ModelFormat.ONNX,
                    inputSize = inputSize,
                    classNames = names,
                    accelerator = if (gpuActive) "NNAPI(GPU)" else "CPU",
                    loadMs = loadMs,
                    gpuActive = gpuActive
                ),
                confThreshold, iouThreshold
            )
        }
    }

    override var confThreshold: Float = confThreshold
    override var iouThreshold: Float = iouThreshold

    private val inputName: String = session.inputNames.first()

    // session 声明在 close 可达处
    private var session: OrtSession? = session

    override fun detect(bitmap: Bitmap): List<BoundingBox> {
        val s = session ?: return emptyList()
        val size = info.inputSize
        val frameW = bitmap.width
        val frameH = bitmap.height

        // letterbox（bitmap 层面）
        val lp = LetterboxParams.compute(frameW, frameH, size)
        val input = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(input)
        canvas.drawColor(Color.rgb(114, 114, 114))
        val scaled = Bitmap.createScaledBitmap(bitmap, lp.newW, lp.newH, true)
        canvas.drawBitmap(scaled, lp.padX, lp.padY, null)
        if (scaled !== bitmap) scaled.recycle()

        // HWC_ARGB → NCHW RGB /255
        val pixels = IntArray(size * size)
        input.getPixels(pixels, 0, size, 0, 0, size, size)
        input.recycle()
        val area = size * size
        val data = FloatArray(3 * area)
        for (i in 0 until area) {
            val p = pixels[i]
            data[i] = (p shr 16 and 0xFF) / 255f
            data[area + i] = (p shr 8 and 0xFF) / 255f
            data[2 * area + i] = (p and 0xFF) / 255f
        }

        val inTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(data), longArrayOf(1, 3, size.toLong(), size.toLong())
        )
        val outputs = s.run(Collections.singletonMap(inputName, inTensor))
        try {
            val outTensor = pickLargestOutput(outputs)
            val shape = outTensor.info.shape
            AppLogger.d(TAG, "onnx output shape=${shape.joinToString("x")}")
            return when (shape.size) {
                3 -> {
                    val d1 = shape[1].toInt()
                    val d2 = shape[2].toInt()
                    val n = (d1.toLong() * d2).toInt()
                    val raw = FloatArray(n)
                    outTensor.floatBuffer.get(raw)
                    // 保证 [C=4+nc, N] 布局
                    val channels: Int
                    val anchors: Int
                    val flat: FloatArray
                    if (d1 <= d2) {
                        channels = d1; anchors = d2; flat = raw
                    } else {
                        channels = d2; anchors = d1
                        flat = YoloPostprocessor.transposeIfNeeded(raw, d1, d2)!!
                    }
                    val cands = YoloPostprocessor.decode(
                        flat, channels, anchors, confThreshold, lp, frameW, frameH
                    )
                    val kept = YoloPostprocessor.nms(cands, iouThreshold)
                    kept.map { c ->
                        BoundingBox(
                            x1 = c.x1, y1 = c.y1, x2 = c.x2, y2 = c.y2,
                            score = c.score,
                            className = info.classNames.getOrElse(c.classId) { "class_${c.classId}" },
                            classId = c.classId
                        )
                    }
                }
                else -> throw ModelException.Inference("不支持的输出维度: ${shape.joinToString("x")}")
            }
        } finally {
            outputs.close()
        }
    }

    private fun pickLargestOutput(outputs: OrtSession.Result): OnnxTensor {
        var best: OnnxTensor? = null
        var bestN = -1L
        for (i in 0 until outputs.size()) {
            val v = outputs.get(i)
            if (v is OnnxTensor) {
                val n = v.info.shape.fold(1L) { a, b -> a * (if (b > 0) b else 1L) }
                if (n > bestN) { bestN = n; best = v }
            }
        }
        return best ?: throw ModelException.Inference("无有效输出张量")
    }

    override fun close() {
        try { session?.close() } catch (_: Exception) {}
        session = null
    }
}

/**
 * TorchScript .pt 引擎（PyTorch Mobile，CPU）。
 * 仅加载经 torch.jit.save / script 导出的 .pt；ultralytics 原始 checkpoint 不支持。
 */
class TorchScriptEngine private constructor(
    private val module: Module,
    override val info: ModelInfo,
    confThreshold: Float,
    iouThreshold: Float
) : DetectorEngine {

    companion object {
        fun create(
            modelFile: File,
            fallbackLabels: List<String>,
            confThreshold: Float,
            iouThreshold: Float
        ): TorchScriptEngine {
            val t0 = System.currentTimeMillis()
            val module = try {
                Module.load(modelFile.absolutePath)
            } catch (e: Exception) {
                throw ModelException.UnsupportedFormat(
                    ModelFormat.TORCHSCRIPT,
                    "TorchScript 加载失败: ${e.message}"
                )
            }
            val loadMs = System.currentTimeMillis() - t0
            return TorchScriptEngine(
                module,
                ModelInfo(
                    fileName = modelFile.name,
                    format = ModelFormat.TORCHSCRIPT,
                    inputSize = 640,
                    classNames = fallbackLabels,
                    accelerator = "CPU(PyTorch)",
                    loadMs = loadMs,
                    gpuActive = false // PyTorch Mobile CPU 后端
                ),
                confThreshold, iouThreshold
            )
        }
    }

    override var confThreshold: Float = confThreshold
    override var iouThreshold: Float = iouThreshold

    override fun detect(bitmap: Bitmap): List<BoundingBox> {
        val size = info.inputSize
        val frameW = bitmap.width
        val frameH = bitmap.height
        val lp = LetterboxParams.compute(frameW, frameH, size)

        val input = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(input)
        canvas.drawColor(Color.rgb(114, 114, 114))
        val scaled = Bitmap.createScaledBitmap(bitmap, lp.newW, lp.newH, true)
        canvas.drawBitmap(scaled, lp.padX, lp.padY, null)
        if (scaled !== bitmap) scaled.recycle()

        val pixels = IntArray(size * size)
        input.getPixels(pixels, 0, size, 0, 0, size, size)
        input.recycle()
        val area = size * size
        val data = FloatArray(3 * area)
        for (i in 0 until area) {
            val p = pixels[i]
            data[i] = (p shr 16 and 0xFF) / 255f
            data[area + i] = (p shr 8 and 0xFF) / 255f
            data[2 * area + i] = (p and 0xFF) / 255f
        }

        val outTensor = try {
            module.forward(
                IValue.from(Tensor.fromBlob(data, longArrayOf(1, 3, size.toLong(), size.toLong())))
            ).toTensor()
        } catch (e: Exception) {
            throw ModelException.Inference("TorchScript forward: ${e.message}", e)
        }

        val shape = outTensor.shape()
        AppLogger.d(TAG, "torchscript output shape=${shape.joinToString("x")}")
        val raw = outTensor.dataAsFloatArray
        val flat: FloatArray
        val channels: Int
        val anchors: Int
        when {
            shape.size == 3 && shape[1] <= shape[2] -> {
                channels = shape[1].toInt(); anchors = shape[2].toInt(); flat = raw
            }
            shape.size == 3 -> {
                channels = shape[2].toInt(); anchors = shape[1].toInt()
                flat = YoloPostprocessor.transposeIfNeeded(raw, shape[1].toInt(), shape[2].toInt())!!
            }
            else -> throw ModelException.Inference("不支持的输出维度: ${shape.joinToString("x")}")
        }

        val cands = YoloPostprocessor.decode(
            flat, channels, anchors, confThreshold, lp, frameW, frameH
        )
        return YoloPostprocessor.nms(cands, iouThreshold).map { c ->
            BoundingBox(
                x1 = c.x1, y1 = c.y1, x2 = c.x2, y2 = c.y2,
                score = c.score,
                className = info.classNames.getOrElse(c.classId) { "class_${c.classId}" },
                classId = c.classId
            )
        }
    }

    override fun close() {
        try { module.destroy() } catch (_: Throwable) {}
    }
}

/**
 * 统一模型加载入口：负责来源读取、格式校验、引擎构建。
 */
class ModelRepository(private val context: Context) {

    sealed class Source {
        data class Asset(val path: String) : Source()
        data class FileSource(val file: File) : Source()
    }

    companion object {
        /** assets 内置模型目录 */
        const val ASSET_MODEL_DIR = "models"

        fun listAssetModels(context: Context): List<String> = try {
            context.assets.list(ASSET_MODEL_DIR)
                ?.filter { it.lowercase().endsWith(".onnx") || it.lowercase().endsWith(".pt") }
                ?.map { "$ASSET_MODEL_DIR/$it" }
                ?: emptyList()
        } catch (_: Exception) { emptyList() }

        fun listFallbackLabels(context: Context): List<String> = try {
            context.assets.open("labels.txt").bufferedReader().readLines()
                .map { it.trim() }.filter { it.isNotEmpty() }
        } catch (_: Exception) { emptyList() }

        /** 读流并检测格式（同时返回全部字节供引擎使用） */
        fun readAndDetect(fileName: String, open: () -> InputStream): Pair<ModelFormat, ByteArray> {
            val bytes = open().use { it.readBytes() }
            if (bytes.isEmpty()) throw ModelException.EmptyFile(fileName)
            val head = bytes.copyOfRange(0, minOf(8, bytes.size))
            var format = ModelFormat.UNKNOWN
            if (head.size >= 2 && head[0] == 0x50.toByte() && head[1] == 0x4B.toByte()) {
                // zip: 解析 entry 名做 TorchScript / pickle 判定
                val names = try {
                    val zin = java.util.zip.ZipInputStream(bytes.inputStream())
                    val list = ArrayList<String>()
                    while (true) {
                        val e = zin.nextEntry ?: break
                        list.add(e.name)
                        zin.closeEntry()
                    }
                    zin.close()
                    list
                } catch (_: Exception) { emptyList() }
                format = ModelFormatDetector.detect(fileName, head, names)
            } else if (fileName.lowercase().endsWith(".onnx")) {
                format = ModelFormat.ONNX
            }
            return format to bytes
        }
    }

    /**
     * 加载模型。失败抛 [ModelException]，UI 层直接展示 message。
     */
    fun load(source: Source, useGpu: Boolean, conf: Float, iou: Float): DetectorEngine {
        val fileName = when (source) {
            is Source.Asset -> source.path.substringAfterLast('/')
            is Source.FileSource -> source.file.name
        }
        val (format, bytes) = when (source) {
            is Source.Asset -> readAndDetect(fileName) { context.assets.open(source.path) }
            is Source.FileSource -> {
                if (!source.file.exists()) throw ModelException.FileNotFound(source.file.absolutePath)
                readAndDetect(fileName) { source.file.inputStream() }
            }
        }

        AppLogger.i(TAG, "load $fileName format=$format bytes=${bytes.size} gpu=$useGpu")

        val labels = listFallbackLabels(context)
        return when (format) {
            ModelFormat.ONNX -> OnnxEngine.create(
                bytes, fileName, labels, useGpu, conf, iou
            )
            ModelFormat.TORCHSCRIPT -> {
                // PyTorch Mobile 需要文件路径：缓存到 filesDir
                val cache = File(context.filesDir, "loaded_ts_model.pt")
                cache.parentFile?.mkdirs()
                cache.writeBytes(bytes)
                TorchScriptEngine.create(cache, labels, conf, iou)
            }
            ModelFormat.PICKLE_CHECKPOINT -> throw ModelException.UnsupportedFormat(
                format,
                "该 .pt 是 Ultralytics 训练 checkpoint（pickle），Android 无法直接反序列化。" +
                    "请在 PC 上执行: yolo export model=<file>.pt format=onnx opset=12，" +
                    "或 format=torchscript 后重新导入。"
            )
            ModelFormat.UNKNOWN -> throw ModelException.UnsupportedFormat(
                format, "文件既不是有效 ONNX，也不是 PyTorch zip 格式。"
            )
        }
    }
}
