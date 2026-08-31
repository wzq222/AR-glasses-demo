package com.example.yolov8

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.GridLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.yolov8.core.log.AppLogger
import com.example.yolov8.core.model.EngineHolder
import com.example.yolov8.core.model.ModelException
import com.example.yolov8.core.model.ModelRepository
import com.example.yolov8.databinding.ActivityTestImageBinding
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * 测试模式：图片批量检测 + 标注。
 * 数据源（优先级）：
 *  1. adb push 目录：/sdcard/Android/data/<pkg>/files/test_images/（无需权限）
 *  2. 应用内置 assets/testset/ 样本
 * 输出：标注图保存到 Pictures/YOLOTest（MediaStore）
 */
class TestImageActivity : AppCompatActivity() {

    private lateinit var binding: ActivityTestImageBinding
    private val executor = Executors.newSingleThreadExecutor()
    private val engineLoading = AtomicBoolean(false)
    private val items = ArrayList<TestItem>()
    private val thumbCache = object : android.util.LruCache<String, Bitmap>(24 * 1024 * 1024) {
        override fun sizeOf(key: String, value: Bitmap) = value.byteCount
    }

    private data class TestItem(val name: String, val isAsset: Boolean, val file: File?)

    private var lastAnnotated: Bitmap? = null
    private var lastName: String = "annotated"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        AppLogger.init(application)
        binding = ActivityTestImageBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)

        val span = if (resources.configuration.orientation ==
            android.content.res.Configuration.ORIENTATION_LANDSCAPE) 4 else 3
        binding.recycler.layoutManager = GridLayoutManager(this, span)
        binding.recycler.adapter = Adapter()

        binding.batchButton.setOnClickListener { runBatch() }
        binding.saveButton.setOnClickListener { saveCurrent() }

        executor.execute { loadItems() }

        // e2e：自动批量运行
        if (intent.getBooleanExtra(EXTRA_AUTO, false)) {
            binding.batchButton.post { runBatch() }
        }
    }

    override fun onSupportNavigateUp(): Boolean {
        finish()
        return true
    }

    private fun loadItems() {
        items.clear()
        val ext = File(getExternalFilesDir(null), "test_images")
        ext.listFiles { f -> f.isFile && f.extension.lowercase() in setOf("jpg", "jpeg", "png") }
            ?.sortedBy { it.name }?.forEach { items.add(TestItem(it.name, false, it)) }
        try {
            assets.list("testset")?.filter {
                it.lowercase().endsWith(".jpg") || it.lowercase().endsWith(".png")
            }?.sorted()?.forEach { items.add(TestItem("testset/$it", true, null)) }
        } catch (_: Exception) {}
        AppLogger.i("Test", "测试图片 ${items.size} 张（外部=${ext.absolutePath}）")
        runOnUiThread { binding.recycler.adapter?.notifyDataSetChanged() }
    }

    private fun decode(name: String, isAsset: Boolean, file: File?, maxDim: Int = 1280): Bitmap? = try {
        val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        if (isAsset) assets.open(name).use { BitmapFactory.decodeStream(it, null, opts) }
        else BitmapFactory.decodeFile(file!!.absolutePath, opts)
        var sample = 1
        while (maxOf(opts.outWidth, opts.outHeight) / (sample * 2) >= maxDim) sample *= 2
        val decodeOpts = BitmapFactory.Options().apply { inSampleSize = sample }
        if (isAsset) assets.open(name).use {
            BitmapFactory.decodeStream(it, null, decodeOpts)
        } else BitmapFactory.decodeFile(file!!.absolutePath, decodeOpts)
    } catch (e: Exception) {
        // 单张解码失败不应导致工作线程崩溃
        AppLogger.w("Test", "decode $name 失败: ${e.message}")
        null
    }

    /**
     * e2e/直接 adb 启动时进程内引擎尚未初始化（通常由 MainActivity 加载）：
     * 这里懒加载第一个内置模型，加载完成后继续原操作。
     */
    private fun ensureEngine(onReady: () -> Unit) {
        if (EngineHolder.current().engine != null) {
            onReady()
            return
        }
        if (!engineLoading.compareAndSet(false, true)) return
        binding.resultText.text = getString(R.string.status_loading_model)
        executor.execute {
            try {
                val path = ModelRepository.listAssetModels(this).firstOrNull()
                    ?: throw ModelException.FileNotFound("assets/models（无内置模型）")
                val repo = ModelRepository(applicationContext)
                val engine = repo.load(ModelRepository.Source.Asset(path), true, 0.25f, 0.45f)
                EngineHolder.set(engine, path, true)
                AppLogger.i(
                    "Test",
                    "懒加载引擎: ${engine.info.fileName} acc=${engine.info.accelerator} " +
                        "load=${engine.info.loadMs}ms"
                )
                runOnUiThread { onReady() }
            } catch (e: ModelException) {
                AppLogger.e("Test", "引擎懒加载失败: ${e.message}")
                runOnUiThread { binding.resultText.text = e.message }
            } catch (e: Exception) {
                AppLogger.e("Test", "引擎懒加载异常", e)
                runOnUiThread { binding.resultText.text = e.message }
            } finally {
                engineLoading.set(false)
            }
        }
    }

    private fun runBatch() {
        ensureEngine { doRunBatch() }
    }

    private fun doRunBatch() {
        val engine = EngineHolder.current().engine
        if (engine == null) {
            binding.resultText.text = getString(R.string.test_no_engine)
            return
        }
        val batch = items.take(BATCH_LIMIT)
        binding.batchButton.isEnabled = false
        binding.resultText.text = getString(R.string.test_running, batch.size)
        AppLogger.i("Test", "批量运行 ${batch.size} 张, model=${engine.info.fileName}")

        executor.execute {
            var ok = 0
            var totalBoxes = 0
            var totalMs = 0L
            val t0 = System.currentTimeMillis()
            for ((idx, item) in batch.withIndex()) {
                val bmp = decode(item.name, item.isAsset, item.file) ?: continue
                try {
                    val s0 = System.currentTimeMillis()
                    val boxes = engine.detect(bmp)
                    val ms = System.currentTimeMillis() - s0
                    totalBoxes += boxes.size
                    totalMs += ms
                    val annotated = AnnotationRenderer.annotateBitmap(
                        bmp, boxes, resources.displayMetrics.density
                    )
                    val base = item.name.substringAfterLast('/').substringBeforeLast('.')
                    val saved = AnnotationRenderer.saveAnnotated(this, annotated, "yolo_$base")
                    if (saved != null) ok++
                    annotated.recycle()
                    AppLogger.d(
                        "Test",
                        "[${idx + 1}/${batch.size}] ${item.name} boxes=${boxes.size} ${ms}ms saved=${saved != null}"
                    )
                    if (idx == 0) {
                        // 第一张结果展示到面板
                        val disp = decode(item.name, item.isAsset, item.file, 1024)
                        if (disp != null) {
                            val ann = AnnotationRenderer.annotateBitmap(
                                disp, boxes, resources.displayMetrics.density
                            )
                            runOnUiThread { showAnnotated(ann, base, boxes.size, ms) }
                        }
                    }
                } catch (e: Exception) {
                    AppLogger.e("Test", "检测失败 ${item.name}", e)
                } finally {
                    bmp.recycle()
                }
            }
            val wall = System.currentTimeMillis() - t0
            AppLogger.i(
                "Test",
                "批量完成: 成功保存=$ok 总框数=$totalBoxes 平均推理=${ if (batch.isEmpty()) 0 else totalMs / batch.size }ms 总耗时=${wall}ms"
            )
            runOnUiThread {
                binding.batchButton.isEnabled = true
                binding.resultText.text = getString(
                    R.string.test_summary, ok, batch.size, totalBoxes,
                    if (batch.isEmpty()) 0 else totalMs / batch.size
                )
            }
        }
    }

    private fun runSingle(item: TestItem) {
        ensureEngine { doRunSingle(item) }
    }

    private fun doRunSingle(item: TestItem) {
        val engine = EngineHolder.current().engine
        if (engine == null) {
            binding.resultText.text = getString(R.string.test_no_engine)
            return
        }
        executor.execute {
            val bmp = decode(item.name, item.isAsset, item.file)
            if (bmp == null) {
                AppLogger.w("Test", "解码失败 ${item.name}")
                return@execute
            }
            try {
                val t0 = System.currentTimeMillis()
                val boxes = engine.detect(bmp)
                val ms = System.currentTimeMillis() - t0
                AppLogger.i("Test", "${item.name} boxes=${boxes.size} ${ms}ms")
                val annotated = AnnotationRenderer.annotateBitmap(
                    bmp, boxes, resources.displayMetrics.density
                )
                runOnUiThread { showAnnotated(annotated, item.name, boxes.size, ms) }
            } catch (e: Exception) {
                AppLogger.e("Test", "检测失败 ${item.name}", e)
                runOnUiThread { binding.resultText.text = e.message }
            } finally {
                bmp.recycle()
            }
        }
    }

    private fun showAnnotated(annotated: Bitmap, name: String, boxes: Int, ms: Long) {
        lastAnnotated?.let { if (!it.isRecycled) it.recycle() }
        lastAnnotated = annotated
        lastName = name.substringAfterLast('/').substringBeforeLast('.')
        binding.annotatedImage.setImageBitmap(annotated)
        binding.resultText.text = getString(R.string.test_single_result, name, boxes, ms)
        binding.resultPanel.visibility = View.VISIBLE
    }

    private fun saveCurrent() {
        val bmp = lastAnnotated ?: return
        executor.execute {
            val path = AnnotationRenderer.saveAnnotated(this, bmp, "yolo_$lastName")
            AppLogger.i("Test", "保存标注图 -> $path")
            runOnUiThread {
                binding.resultText.text =
                    if (path != null) getString(R.string.test_saved, path)
                    else getString(R.string.test_save_failed)
            }
        }
    }

    private inner class Adapter : RecyclerView.Adapter<VH>() {
        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
            val v = LayoutInflater.from(parent.context)
                .inflate(R.layout.item_test_thumb, parent, false)
            return VH(v)
        }

        override fun getItemCount(): Int = items.size

        override fun onBindViewHolder(holder: VH, position: Int) {
            val item = items[position]
            holder.name.text = item.name
            val cached = thumbCache.get(item.name)
            if (cached != null) {
                holder.thumb.setImageBitmap(cached)
            } else {
                holder.thumb.setImageResource(R.drawable.ic_launcher)
                executor.execute {
                    val bmp = decode(item.name, item.isAsset, item.file, 256) ?: return@execute
                    thumbCache.put(item.name, bmp)
                    runOnUiThread {
                        if (items.getOrNull(position) === item) {
                            holder.thumb.setImageBitmap(bmp)
                        }
                    }
                }
            }
            holder.itemView.setOnClickListener { runSingle(item) }
        }
    }

    private class VH(v: View) : RecyclerView.ViewHolder(v) {
        val thumb: ImageView = v.findViewById(R.id.thumb)
        val name: TextView = v.findViewById(R.id.thumbName)
    }

    companion object {
        const val EXTRA_AUTO = "auto_run"
        const val BATCH_LIMIT = 20
    }
}