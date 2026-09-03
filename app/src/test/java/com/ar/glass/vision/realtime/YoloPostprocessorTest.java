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

    @Test
    public void overloadUsesPublicDefaultThresholds() {
        float[][] prediction = prediction(
                new float[]{100f, 105f, 300f, 400f},
                new float[]{100f, 105f, 300f, 400f},
                new float[]{100f, 100f, 20f, 20f},
                new float[]{100f, 100f, 20f, 20f},
                new float[]{0.90f, 0.10f, 0.19f, 0.20f},
                new float[]{0.10f, 0.80f, 0.10f, 0.05f});

        List<Detection> detections = YoloPostprocessor.process(
                prediction, 640, 640, 1f, 0f, 0f);

        assertEquals(0.20f, YoloPostprocessor.DEFAULT_CONFIDENCE_THRESHOLD, EPSILON);
        assertEquals(0.45f, YoloPostprocessor.DEFAULT_NMS_IOU_THRESHOLD, EPSILON);
        assertEquals(2, detections.size());
        assertEquals(0.90f, detections.get(0).getConfidence(), EPSILON);
        assertEquals(0.20f, detections.get(1).getConfidence(), EPSILON);
    }

    @Test
    public void capsAndDeterministicallyOrdersLargeCandidateOutput() {
        int candidateCount = 34_000;
        float[][] prediction = new float[6][candidateCount];
        for (int index = 0; index < candidateCount; index++) {
            prediction[0][index] = index * 3f + 1f;
            prediction[1][index] = 1f;
            prediction[2][index] = 1f;
            prediction[3][index] = 1f;
            prediction[4][index] = 0.90f;
            prediction[5][index] = 0.10f;
        }

        List<Detection> firstRun = YoloPostprocessor.process(
                prediction, 120_000, 10, 1f, 0f, 0f);
        List<Detection> secondRun = YoloPostprocessor.process(
                prediction, 120_000, 10, 1f, 0f, 0f);

        assertEquals(1_000, YoloPostprocessor.DEFAULT_PRE_NMS_TOP_K);
        assertEquals(100, YoloPostprocessor.DEFAULT_MAX_DETECTIONS);
        assertEquals(YoloPostprocessor.DEFAULT_MAX_DETECTIONS, firstRun.size());
        assertEquals(firstRun.size(), secondRun.size());
        for (int index = 0; index < firstRun.size(); index++) {
            assertEquals(firstRun.get(index).getLeft(), secondRun.get(index).getLeft(), EPSILON);
            assertEquals(firstRun.get(index).getConfidence(),
                    secondRun.get(index).getConfidence(), EPSILON);
            assertEquals(firstRun.get(index).getClassId(), secondRun.get(index).getClassId());
        }
        assertEquals(0.5f, firstRun.get(0).getLeft(), EPSILON);
        assertEquals(297.5f, firstRun.get(99).getLeft(), EPSILON);
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
