package com.example.yolov8

import com.example.yolov8.core.log.AppLogger
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AppLoggerTest {

    private class CollectSink : AppLogger.Sink {
        val lines = mutableListOf<Pair<AppLogger.Level, String>>()
        override fun onLog(level: AppLogger.Level, tag: String, message: String, tr: Throwable?) {
            lines.add(level to message)
        }
    }

    @Test
    fun `debug 模式记录所有级别`() {
        val sink = CollectSink()
        AppLogger.productionMode = false
        AppLogger.addSink(sink)
        try {
            AppLogger.d("T", "dbg")
            AppLogger.i("T", "info")
            AppLogger.w("T", "warn")
            AppLogger.e("T", "err")
            assertEquals(4, sink.lines.size)
        } finally {
            AppLogger.clearRecent()
        }
    }

    @Test
    fun `production 模式丢弃 DEBUG INFO 保留 WARN ERROR`() {
        val sink = CollectSink()
        AppLogger.productionMode = true
        AppLogger.addSink(sink)
        try {
            AppLogger.d("T", "dbg")
            AppLogger.i("T", "info")
            AppLogger.w("T", "warn")
            AppLogger.e("T", "err")
            assertEquals(2, sink.lines.size)
            assertEquals(AppLogger.Level.WARN, sink.lines[0].first)
            assertEquals(AppLogger.Level.ERROR, sink.lines[1].first)
            assertTrue(sink.lines[0].second.contains("warn"))
        } finally {
            AppLogger.productionMode = false
            AppLogger.clearRecent()
        }
    }

    @Test
    fun `环形缓冲记录最近日志`() {
        AppLogger.productionMode = false
        try {
            AppLogger.i("Ring", "hello-ring")
            val lines = AppLogger.recentLines()
            assertTrue(lines.any { it.contains("hello-ring") })
        } finally {
            AppLogger.clearRecent()
        }
    }
}