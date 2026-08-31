package com.example.yolov8

/**
 * 目标检测结果框。
 * 坐标已归一化到 [0, 1]，用于 OverlayView 按视图尺寸绘制。
 */
data class BoundingBox(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val score: Float,
    val className: String,
    val classId: Int = -1
)