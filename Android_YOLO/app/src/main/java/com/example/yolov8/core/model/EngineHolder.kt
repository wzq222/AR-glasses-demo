package com.example.yolov8.core.model

import java.util.concurrent.atomic.AtomicReference

/**
 * 进程级引擎单例：MainActivity 与 TestImageActivity 共享同一引擎实例，
 * 避免重复加载模型（加载约 0.5~3s）。
 */
object EngineHolder {

    data class State(
        val engine: DetectorEngine?,
        val sourcePath: String,
        val useGpu: Boolean,
        val error: String? = null
    )

    private val state = AtomicReference<State>(State(null, "", false, null))

    fun current(): State = state.get()

    fun set(engine: DetectorEngine?, sourcePath: String, useGpu: Boolean, error: String? = null) {
        val old = state.getAndSet(State(engine, sourcePath, useGpu, error))
        try { old.engine?.close() } catch (_: Exception) {}
    }
}