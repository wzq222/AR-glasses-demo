package com.ar.glass.vision.fastener;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class FastenerInspection {
    private final String id;
    private final float x;
    private final float y;
    private final float width;
    private final float height;
    private final float detectorConfidence;
    private final List<VisionPoint> keypoints;
    private final GeometryDecision decision;

    public FastenerInspection(
            String id,
            float x,
            float y,
            float width,
            float height,
            float detectorConfidence,
            List<VisionPoint> keypoints,
            GeometryDecision decision) {
        this.id = id;
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.detectorConfidence = detectorConfidence;
        this.keypoints = Collections.unmodifiableList(new ArrayList<>(keypoints));
        this.decision = decision;
    }

    public String getId() { return id; }
    public float getX() { return x; }
    public float getY() { return y; }
    public float getWidth() { return width; }
    public float getHeight() { return height; }
    public float getDetectorConfidence() { return detectorConfidence; }
    public List<VisionPoint> getKeypoints() { return keypoints; }
    public GeometryDecision getDecision() { return decision; }
}
