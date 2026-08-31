package com.example.yolov8

import android.Manifest
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.example.yolov8.core.log.AppLogger
import com.example.yolov8.core.model.DetectorEngine
import com.example.yolov8.core.model.EngineHolder
import com.example.yolov8.core.model.ModelException
import com.example.yolov8.core.model.ModelRepository
import com.example.yolov8.core.permissions.PermissionManager
import com.example.yolov8.core.log.LogActivity
import com.example.yolov8.databinding.ActivityMainBinding
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val bgExecutor = Executors.newSingleThreadExecutor()
    private val engineLoading = AtomicBoolean(false)

    private var useGpu = true
    private var modelPath: String? = null // asset path 或绝对路径

    // 虚拟相机轮播
    private val virtualHandler = android.os.Handler(android.os.Looper.getMainLooper())
    private var virtualFrames: List<FrameSource> = emptyList()
    private var virtualIndex = 0
    private var virtualRunning = false
    private var currentVirtualBitmap: Bitmap? = null

    private data class FrameSource(val name: String, val isAsset: Boolean)

    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else {
                binding.statusText.text = getString(R.string.status_camera_denied)
            }
        }

    private val modelPicker =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
            if (uri != null) importAndLoadModel(uri)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        AppLogger.init(application)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setSupportActionBar(binding.toolbar)

        setupControls()
        pickDefaultModel()

        // e2e 测试入口：intent extra 直接进虚拟相机模式（不请求相机权限）
        if (intent.getBooleanExtra(EXTRA_VIRTUAL, false)) {
            setMode(MODE_VIRTUAL)
        } else {
            requestCameraIfNeeded()
        }
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.menu_main, menu)
        menu.findItem(R.id.action_logs).isVisible = !AppLogger.productionMode
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean = when (item.itemId) {
        R.id.action_logs -> { startActivity(Intent(this, LogActivity::class.java)); true }
        R.id.action_pick_model -> {
            modelPicker.launch(arrayOf("*/*")); true
        }
        R.id.action_test_images -> {
            startActivity(Intent(this, TestImageActivity::class.java)); true
        }
        else -> super.onOptionsItemSelected(item)
    }

    // ---------- 控件与模式 ----------

    private fun setupControls() {
        binding.gpuSwitch.setOnCheckedChangeListener { _, checked ->
            useGpu = checked
            AppLogger.i("Main", "GPU -> $checked")
            reloadEngine()
        }
        binding.confSlider.addOnChangeListener { _, value, fromUser ->
            if (fromUser) {
                binding.confValue.text = String.format("%.2f", value)
                EngineHolder.current().engine?.confThreshold = value
            }
        }
        binding.modeCamera.setOnClickListener { setMode(MODE_CAMERA) }
        binding.modeVirtual.setOnClickListener { setMode(MODE_VIRTUAL) }
        binding.confValue.text = String.format("%.2f", binding.confSlider.value)
    }

    private fun pickDefaultModel() {
        val assets = ModelRepository.listAssetModels(this)
        modelPath = assets.firstOrNull()
        if (modelPath == null) {
            binding.statusText.text = getString(R.string.status_no_model)
            AppLogger.w("Main", "assets/models 下没有内置模型")
        }
        reloadEngine()
    }

    private fun importAndLoadModel(uri: Uri) {
        bgExecutor.execute {
            try {
                // 复制到应用私有目录，避免持久化 SAF 权限问题
                val modelsDir = File(filesDir, "models").apply { mkdirs() }
                val name = queryDisplayName(uri) ?: "imported_model"
                val dst = File(modelsDir, name)
                contentResolver.openInputStream(uri)?.use { input ->
                    dst.outputStream().use { output -> input.copyTo(output) }
                } ?: throw ModelException.FileNotFound(uri.toString())
                AppLogger.i("Main", "导入模型 -> ${dst.absolutePath}")
                modelPath = dst.absolutePath
                runOnUiThread { reloadEngine() }
            } catch (e: Exception) {
                AppLogger.e("Main", "导入模型失败", e)
                runOnUiThread {
                    binding.statusText.text = getString(R.string.status_import_failed, e.message)
                }
            }
        }
    }

    private fun queryDisplayName(uri: Uri): String? =
        contentResolver.query(uri, null, null, null, null)?.use { c ->
            val idx = c.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
            if (idx >= 0 && c.moveToFirst()) c.getString(idx) else null
        }

    /** 后台（重）加载引擎 */
    private fun reloadEngine() {
        val path = modelPath ?: return
        if (!engineLoading.compareAndSet(false, true)) return
        binding.statusText.text = getString(R.string.status_loading_model)
        bgExecutor.execute {
            try {
                val repo = ModelRepository(applicationContext)
                val source = if (path.startsWith("models/"))
                    ModelRepository.Source.Asset(path)
                else ModelRepository.Source.FileSource(File(path))
                val engine = repo.load(source, useGpu, binding.confSlider.value, 0.45f)
                EngineHolder.set(engine, path, useGpu)
                AppLogger.i(
                    "Main",
                    "engine ready: ${engine.info.fileName} acc=${engine.info.accelerator} " +
                        "in=${engine.info.inputSize} classes=${engine.info.classNames.size} " +
                        "load=${engine.info.loadMs}ms"
                )
                runOnUiThread {
                    binding.statusText.text = getString(
                        R.string.status_ready,
                        engine.info.fileName,
                        engine.info.accelerator,
                        engine.info.inputSize
                    )
                    if (virtualRunning) {
                        // 类别集可能变化，清空旧框
                        binding.overlay.clear()
                    }
                    // GPU 加速结果弹窗（用户开启 GPU 时，成功/失败都提示）
                    if (useGpu) showGpuResultDialog(engine.info)
                }
            } catch (e: ModelException) {
                AppLogger.e("Main", "模型加载失败: ${e.message}")
                EngineHolder.set(null, path, useGpu, e.message)
                runOnUiThread { binding.statusText.text = e.message }
            } catch (e: Exception) {
                AppLogger.e("Main", "模型加载异常", e)
                EngineHolder.set(null, path, useGpu, e.message)
                runOnUiThread { binding.statusText.text = e.message }
            } finally {
                engineLoading.set(false)
            }
        }
    }

    // ---------- 相机模式 ----------

    private fun requestCameraIfNeeded() {
        if (PermissionManager.hasCamera(this)) {
            startCamera()
        } else {
            PermissionManager.requestWithRationale(
                this, cameraPermissionLauncher, Manifest.permission.CAMERA,
                getString(R.string.perm_camera_title),
                getString(R.string.perm_camera_rationale)
            )
        }
    }

    private fun startCamera() {
        if (binding.modeVirtual.isChecked) return // 虚拟模式不绑相机
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val provider = future.get()
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(binding.previewView.surfaceProvider)
                }
                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also {
                        it.setAnalyzer(bgExecutor, YoloAnalysis({ engineOrNull() }) { boxes, elapsed, fw, fh ->
                            runOnUiThread {
                                binding.overlay.setResults(boxes, fw, fh)
                                binding.fpsText.text =
                                    getString(R.string.fps_label, boxes.size, elapsed)
                            }
                        })
                    }
                provider.unbindAll()
                provider.bindToLifecycle(
                    this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis
                )
                AppLogger.i("Main", "相机已绑定")
            } catch (e: Exception) {
                AppLogger.e("Main", "相机绑定失败", e)
                binding.statusText.text = getString(R.string.status_camera_failed, e.message)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun engineOrNull(): DetectorEngine? = EngineHolder.current().engine

    /** GPU 加速结果弹窗：成功提示已启用，失败（回退 CPU）提示原因 */
    private fun showGpuResultDialog(info: com.example.yolov8.core.model.ModelInfo) {
        val (title, msg, icon) = if (info.gpuActive) {
            Triple(
                getString(R.string.gpu_dialog_success_title),
                getString(R.string.gpu_dialog_success_msg, info.accelerator),
                android.R.drawable.ic_dialog_info
            )
        } else {
            Triple(
                getString(R.string.gpu_dialog_fallback_title),
                getString(R.string.gpu_dialog_fallback_msg, info.accelerator),
                android.R.drawable.ic_dialog_alert
            )
        }
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle(title)
            .setMessage(msg)
            .setIcon(icon)
            .setPositiveButton(android.R.string.ok, null)
            .show()
        AppLogger.i("Main", "GPU 弹窗: active=${info.gpuActive} acc=${info.accelerator}")
    }

    // ---------- 虚拟相机模式 ----------

    private fun collectVirtualFrames(): List<FrameSource> {
        val list = ArrayList<FrameSource>()
        try {
            assets.list("testset")?.filter {
                it.lowercase().endsWith(".jpg") || it.lowercase().endsWith(".png")
            }?.forEach { list.add(FrameSource("testset/$it", true)) }
        } catch (_: Exception) {}
        val ext = File(getExternalFilesDir(null), "test_images")
        ext.listFiles { f -> f.isFile && f.extension.lowercase() in setOf("jpg", "jpeg", "png") }
            ?.sortedBy { it.name }?.forEach { list.add(FrameSource(it.absolutePath, false)) }
        return list
    }

    private fun setMode(mode: String) {
        val virtual = mode == MODE_VIRTUAL
        AppLogger.i("Main", "mode -> $mode")
        if (virtual) {
            binding.modeVirtual.isChecked = true
            binding.modeCamera.isChecked = false
            startVirtual()
        } else {
            binding.modeCamera.isChecked = true
            binding.modeVirtual.isChecked = false
            stopVirtual()
            if (PermissionManager.hasCamera(this)) startCamera()
            else requestCameraIfNeeded()
        }
        binding.previewView.visibility = if (virtual) View.GONE else View.VISIBLE
        binding.virtualImage.visibility = if (virtual) View.VISIBLE else View.GONE
        // 映射模式与显示控件同步：虚拟图片 fitCenter，相机预览 fillCenter
        binding.overlay.setFillCenter(!virtual)
    }

    private fun startVirtual() {
        virtualFrames = collectVirtualFrames()
        if (virtualFrames.isEmpty()) {
            binding.statusText.text = getString(R.string.status_no_test_images)
            return
        }
        AppLogger.i("Main", "虚拟相机帧源 ${virtualFrames.size} 张")
        virtualRunning = true
        virtualIndex = 0
        tickVirtual()
    }

    private fun stopVirtual() {
        virtualRunning = false
        virtualHandler.removeCallbacksAndMessages(null)
        binding.overlay.clear()
        currentVirtualBitmap?.let { if (!it.isRecycled) it.recycle() }
        currentVirtualBitmap = null
    }

    private fun tickVirtual() {
        if (!virtualRunning) return
        val src = virtualFrames.getOrNull(virtualIndex) ?: return
        bgExecutor.execute {
            val bmp = loadFrame(src)
            if (bmp == null) {
                AppLogger.w("Virtual", "帧加载失败: ${src.name}")
            } else {
                val engine = EngineHolder.current().engine
                val boxes = if (engine != null) {
                    try { engine.detect(bmp) } catch (e: Exception) {
                        AppLogger.w("Virtual", "检测失败 ${src.name}: ${e.message}")
                        emptyList()
                    }
                } else emptyList()
                runOnUiThread {
                    if (!virtualRunning) { bmp.recycle(); return@runOnUiThread }
                    currentVirtualBitmap?.let { if (!it.isRecycled && it !== bmp) it.recycle() }
                    currentVirtualBitmap = bmp
                    binding.virtualImage.setImageBitmap(bmp)
                    binding.overlay.setResults(boxes, bmp.width, bmp.height)
                    binding.fpsText.text = getString(R.string.virtual_label, src.name.substringAfterLast('/'), boxes.size)
                }
            }
            virtualIndex = (virtualIndex + 1) % virtualFrames.size.coerceAtLeast(1)
            virtualHandler.postDelayed({ tickVirtual() }, VIRTUAL_INTERVAL_MS)
        }
    }

    private fun loadFrame(src: FrameSource): Bitmap? = try {
        if (src.isAsset) assets.open(src.name).use { BitmapFactory.decodeStream(it) }
        else BitmapFactory.decodeFile(src.name)
    } catch (e: Exception) {
        AppLogger.w("Virtual", "decode ${src.name}: ${e.message}")
        null
    }

    override fun onDestroy() {
        super.onDestroy()
        stopVirtual()
        bgExecutor.shutdown()
    }

    companion object {
        const val MODE_CAMERA = "camera"
        const val MODE_VIRTUAL = "virtual"
        const val EXTRA_VIRTUAL = "start_virtual"
        private const val VIRTUAL_INTERVAL_MS = 900L
    }
}