package com.ar.glass.vision.realtime;

import java.util.List;

public final class DetectionHitTester {
    private DetectionHitTester() {}

    public static Detection smallestContainingFillCenter(
            List<Detection> detections,
            int imageWidth,
            int imageHeight,
            int previewWidth,
            int previewHeight,
            float touchX,
            float touchY) {
        if (detections == null) {
            throw new IllegalArgumentException("detections are required");
        }
        PreviewCoordinateMapper mapper = PreviewCoordinateMapper.fillCenter(
                imageWidth, imageHeight, previewWidth, previewHeight);
        Detection selected = null;
        float selectedArea = Float.POSITIVE_INFINITY;
        for (Detection detection : detections) {
            PreviewCoordinateMapper.MappedRect box = mapper.map(
                    detection.getLeft(), detection.getTop(),
                    detection.getRight(), detection.getBottom());
            if (touchX < box.getLeft() || touchX > box.getRight()
                    || touchY < box.getTop() || touchY > box.getBottom()) {
                continue;
            }
            float area = Math.max(0f, detection.getRight() - detection.getLeft())
                    * Math.max(0f, detection.getBottom() - detection.getTop());
            if (area < selectedArea) {
                selected = detection;
                selectedArea = area;
            }
        }
        return selected;
    }
}
