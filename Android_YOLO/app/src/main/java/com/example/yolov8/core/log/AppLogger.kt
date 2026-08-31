package com.example.yolov8.core.log

import android.app.Application
import android.util.Log
import com.example.yolov8.BuildConfig
import java.io.File
import java.text.SimpleDateFormat
import java.util.ArrayDeque
import java.util.Date
import java.util.Locale
import java.util.concurrent.CopyOnWriteArrayList

/**
 * 统一日志门面：
 *  - Debug 构建：Logcat + 内存环形缓冲 + 文件（app 外部 files/logs/，可 adb pull）
 *  - Release（生产）构建：丢弃 V/D/I（测试日志全部隐藏），仅保留 W/E 到 Logcat
 *  - 支持注入自定义 Sink 便于单元测试
 */
object AppLogger {

    const val PREFIX = "YOLO"

    enum class Level { DEBUG, INFO, WARN, ERROR }

    interface Sink {
        fun onLog(level: Level, tag: String, message: String, tr: Throwable?)
    }

    @Volatile
    var productionMode: Boolean = !BuildConfig.DEBUG
        internal set // 测试可注入；生产由 init 按 BuildType 决定

    private val sinks = CopyOnWriteArrayList<Sink>()
    private val recent = ArrayDeque<String>()
    private const val RING_CAPACITY = 600
    private val timeFmt = SimpleDateFormat("MM-dd HH:mm:ss.SSS", Locale.US)

    private const val MAX_LOG_BYTES = 1L * 1024 * 1024

    fun init(app: Application) {
        if (sinks.isNotEmpty()) return // 只初始化一次
        productionMode = !BuildConfig.DEBUG
        addSink(LogcatSink)
        if (!productionMode) {
            val dir = app.getExternalFilesDir("logs") ?: File(app.filesDir, "logs")
            addSink(FileSink(dir))
        }
        i("App", "logger init, production=$productionMode")
    }

    /** 测试或特殊场景注入额外 sink */
    fun addSink(sink: Sink) {
        sinks.add(sink)
    }

    fun d(tag: String, msg: String, tr: Throwable? = null) = dispatch(Level.DEBUG, tag, msg, tr)
    fun i(tag: String, msg: String, tr: Throwable? = null) = dispatch(Level.INFO, tag, msg, tr)
    fun w(tag: String, msg: String, tr: Throwable? = null) = dispatch(Level.WARN, tag, msg, tr)
    fun e(tag: String, msg: String, tr: Throwable? = null) = dispatch(Level.ERROR, tag, msg, tr)

    private fun dispatch(level: Level, tag: String, msg: String, tr: Throwable?) {
        if (productionMode && (level == Level.DEBUG || level == Level.INFO)) return
        val line = "${timeFmt.format(Date())} [${level.name.first()}]/$PREFIX.$tag: $msg" +
            (tr?.let { " -> ${it.javaClass.simpleName}: ${it.message}" } ?: "")
        for (s in sinks) {
            try { s.onLog(level, "$PREFIX.$tag", msg, tr) } catch (_: Exception) {}
        }
        record(line)
    }

    @Synchronized
    private fun record(line: String) {
        if (recent.size >= RING_CAPACITY) recent.removeFirst()
        recent.addLast(line)
    }

    /** 内存中最近日志（日志查看器用） */
    @Synchronized
    fun recentLines(): List<String> = recent.toList()

    @Synchronized
    fun clearRecent() = recent.clear()

    object LogcatSink : Sink {
        override fun onLog(level: Level, tag: String, message: String, tr: Throwable?) {
            when (level) {
                Level.DEBUG -> Log.d(tag, message, tr)
                Level.INFO -> Log.i(tag, message, tr)
                Level.WARN -> Log.w(tag, message, tr)
                Level.ERROR -> Log.e(tag, message, tr)
            }
        }
    }

    /** 滚动文件日志（仅 debug 构建注册） */
    class FileSink(private val dir: File) : Sink {

        private val file: File get() = File(dir, "yolo.log")
        private val lock = Any()

        init { dir.mkdirs() }

        override fun onLog(level: Level, tag: String, message: String, tr: Throwable?) {
            val line = "${timeFmt.format(Date())} [${level.name.first()}]/$tag: $message" +
                (tr?.let { android.util.Log.getStackTraceString(it) } ?: "") + "\n"
            synchronized(lock) {
                try {
                    if (file.exists() && file.length() > MAX_LOG_BYTES) {
                        val old = File(dir, "yolo.old.log")
                        old.delete()
                        file.renameTo(old)
                    }
                    file.appendText(line)
                } catch (_: Exception) {}
            }
        }
    }
}
