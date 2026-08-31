package com.ar.glass.vision;

import android.graphics.Bitmap;

import java.util.List;

/**
 * 一轮 YOLO 检测结果（经 EventBus 发往 UI）：
 *  - preview 为带所有权交接的预览图（UI 显示新图时负责 recycle 旧图）
 *  - detections 坐标相对原始照片归一化
 *  - 失败轮次 preview 为 null 且 detections 为空，用 error 描述原因
 */
public class DetectResult {

    public final Bitmap preview;
    public final List<YoloDetector.Detection> detections;
    public final int frameW;
    public final int frameH;
    public final long inferMs;
    public final String fileName;
    public final String error;

    public DetectResult(Bitmap preview, List<YoloDetector.Detection> detections,
                        int frameW, int frameH, long inferMs, String fileName) {
        this(preview, detections, frameW, frameH, inferMs, fileName, null);
    }

    public DetectResult(String error) {
        this(null, null, 0, 0, 0, null, error);
    }

    private DetectResult(Bitmap preview, List<YoloDetector.Detection> detections,
                         int frameW, int frameH, long inferMs, String fileName, String error) {
        this.preview = preview;
        this.detections = detections;
        this.frameW = frameW;
        this.frameH = frameH;
        this.inferMs = inferMs;
        this.fileName = fileName;
        this.error = error;
    }

    public boolean isSuccess() {
        return error == null && preview != null;
    }
}
