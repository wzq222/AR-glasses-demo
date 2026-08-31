package com.ar.glass.vision.realtime;

import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;

public class YoloPostprocessorTest {
    private static final float EPSILON = 0.0001f;

    @Test
    public void usesMaximumClassScoreForEachCandidate() {
        float[][] prediction = prediction(
                new float[]{100f, 300f},
                new float[]{100f, 300f},
                new float[]{40f, 40f},
                new float[]{20f, 20f},
                new float[]{0.30f, 0.70f},
                new float[]{0.80f, 0.20f});

        List<Detection> detections = process(prediction, 0.20f, 0.45f);

        assertEquals(2, detections.size());
        assertEquals(1, detections.get(0).getClassId());
        assertEquals(0.80f, detections.get(0).getConfidence(), EPSILON);
        assertEquals(0, detections.get(1).getClassId());
        assertEquals(0.70f, detections.get(1).getConfidence(), EPSILON);
    }

    @Test
    public void filtersScoresBelowConfidenceThreshold() {
        float[][] prediction = prediction(
                new float[]{100f, 300f},
                new float[]{100f, 300f},
                new float[]{20f, 20f},
                new float[]{20f, 20f},
                new float[]{0.19f, 0.20f},
                new float[]{0.10f, 0.05f});

        List<Detection> detections = process(prediction, 0.20f, 0.45f);

        assertEquals(1, detections.size());
        assertEquals(0.20f, detections.get(0).getConfidence(), EPSILON);
        assertEquals(290f, detections.get(0).getLeft(), EPSILON);
    }

    @Test
    public void reversesLetterboxCoordinatesAndClipsToOriginalImage() {
        float[][] prediction = prediction(
                new float[]{10f},
                new float[]{130f},
                new float[]{40f},
                new float[]{80f},
                new float[]{0.90f},
                new float[]{0.10f});

        List<Detection> detections = YoloPostprocessor.process(
                prediction, 1280, 720, 0.5f, 0f, 140f, 0.20f, 0.45f);

        assertEquals(1, detections.size());
        Detection detection = detections.get(0);
        assertEquals(0f, detection.getLeft(), EPSILON);
        assertEquals(0f, detection.getTop(), EPSILON);
        assertEquals(60f, detection.getRight(), EPSILON);
        assertEquals(60f, detection.getBottom(), EPSILON);
    }

    @Test
    public void appliesClassAgnosticNmsAtConfiguredIou() {
        float[][] prediction = prediction(
                new float[]{100f, 105f, 300f},
                new float[]{100f, 105f, 300f},
                new float[]{100f, 100f, 40f},
                new float[]{100f, 100f, 40f},
                new float[]{0.90f, 0.10f, 0.10f},
                new float[]{0.10f, 0.80f, 0.70f});

        List<Detection> detections = process(prediction, 0.20f, 0.45f);

        assertEquals(2, detections.size());
        assertEquals(0.90f, detections.get(0).getConfidence(), EPSILON);
        assertEquals(0, detections.get(0).getClassId());
        assertEquals(0.70f, detections.get(1).getConfidence(), EPSILON);
        assertEquals(1, detections.get(1).getClassId());
    }

    @Test(expected = UnsupportedOperationException.class)
    public void returnsAnImmutableDetectionList() {
        List<Detection> detections = process(prediction(
                new float[]{100f},
                new float[]{100f},
                new float[]{20f},
                new float[]{20f},
                new float[]{0.90f},
                new float[]{0.10f}), 0.20f, 0.45f);

        detections.add(detections.get(0));
    }

    private static List<Detection> process(
            float[][] prediction, float confidenceThreshold, float nmsThreshold) {
        return YoloPostprocessor.process(
                prediction, 640, 640, 1f, 0f, 0f, confidenceThreshold, nmsThreshold);
    }

    private static float[][] prediction(float[]... rows) {
        return rows;
    }
}
