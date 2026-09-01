package com.ar.glass.vision.realtime;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.nio.FloatBuffer;

public final class YoloPostprocessor {
    public static final float DEFAULT_CONFIDENCE_THRESHOLD = 0.20f;
    public static final float DEFAULT_NMS_IOU_THRESHOLD = 0.45f;
    public static final int DEFAULT_PRE_NMS_TOP_K = 1_000;
    public static final int DEFAULT_MAX_DETECTIONS = 100;

    private static final int OUTPUT_ROWS = 6;

    private YoloPostprocessor() {
    }

    public static List<Detection> process(
            float[][] prediction,
            int originalWidth,
            int originalHeight,
            float scale,
            float padX,
            float padY) {
        return process(
                prediction,
                originalWidth,
                originalHeight,
                scale,
                padX,
                padY,
                DEFAULT_CONFIDENCE_THRESHOLD,
                DEFAULT_NMS_IOU_THRESHOLD);
    }

    public static List<Detection> process(
            float[][] prediction,
            int originalWidth,
            int originalHeight,
            float scale,
            float padX,
            float padY,
            float confidenceThreshold,
            float nmsIouThreshold) {
        int candidateCount = validateArguments(
                prediction,
                originalWidth,
                originalHeight,
                scale,
                padX,
                padY,
                confidenceThreshold,
                nmsIouThreshold);

        return processValidated(
                new PredictionReader() {
                    @Override
                    public float get(int row, int column) {
                        return prediction[row][column];
                    }
                },
                candidateCount,
                originalWidth,
                originalHeight,
                scale,
                padX,
                padY,
                confidenceThreshold,
                nmsIouThreshold);
    }

    public static List<Detection> process(
            FloatBuffer prediction,
            int candidateCount,
            int originalWidth,
            int originalHeight,
            float scale,
            float padX,
            float padY) {
        if (prediction == null || candidateCount < 0
                || prediction.capacity() < OUTPUT_ROWS * candidateCount) {
            throw new IllegalArgumentException("flat prediction buffer is too small");
        }
        validateGeometry(
                originalWidth,
                originalHeight,
                scale,
                padX,
                padY,
                DEFAULT_CONFIDENCE_THRESHOLD,
                DEFAULT_NMS_IOU_THRESHOLD);
        return processValidated(
                new PredictionReader() {
                    @Override
                    public float get(int row, int column) {
                        return prediction.get(row * candidateCount + column);
                    }
                },
                candidateCount,
                originalWidth,
                originalHeight,
                scale,
                padX,
                padY,
                DEFAULT_CONFIDENCE_THRESHOLD,
                DEFAULT_NMS_IOU_THRESHOLD);
    }

    public static List<Detection> process(
            FloatBuffer prediction,
            int candidateCount,
            int originalWidth,
            int originalHeight,
            float scale,
            float padX,
            float padY,
            float confidenceThreshold,
            float nmsIouThreshold) {
        if (prediction == null || candidateCount < 0
                || prediction.capacity() < OUTPUT_ROWS * candidateCount) {
            throw new IllegalArgumentException("flat prediction buffer is too small");
        }
        validateGeometry(
                originalWidth,
                originalHeight,
                scale,
                padX,
                padY,
                confidenceThreshold,
                nmsIouThreshold);
        return processValidated(
                new PredictionReader() {
                    @Override
                    public float get(int row, int column) {
                        return prediction.get(row * candidateCount + column);
                    }
                },
                candidateCount,
                originalWidth,
                originalHeight,
                scale,
                padX,
                padY,
                confidenceThreshold,
                nmsIouThreshold);
    }

    private static List<Detection> processValidated(
            PredictionReader prediction,
            int candidateCount,
            int originalWidth,
            int originalHeight,
            float scale,
            float padX,
            float padY,
            float confidenceThreshold,
            float nmsIouThreshold) {
        List<Detection> candidates = new ArrayList<>();
        for (int index = 0; index < candidateCount; index++) {
            float class0 = prediction.get(4, index);
            float class1 = prediction.get(5, index);
            int classId = class1 > class0 ? 1 : 0;
            float confidence = classId == 1 ? class1 : class0;
            if (!isFinite(confidence) || confidence < confidenceThreshold) {
                continue;
            }

            float centerX = prediction.get(0, index);
            float centerY = prediction.get(1, index);
            float width = prediction.get(2, index);
            float height = prediction.get(3, index);
            if (!isFinite(centerX) || !isFinite(centerY)
                    || !isFinite(width) || !isFinite(height)
                    || width <= 0f || height <= 0f) {
                continue;
            }

            float left = clip((centerX - width / 2f - padX) / scale, 0f, originalWidth);
            float top = clip((centerY - height / 2f - padY) / scale, 0f, originalHeight);
            float right = clip((centerX + width / 2f - padX) / scale, 0f, originalWidth);
            float bottom = clip((centerY + height / 2f - padY) / scale, 0f, originalHeight);
            if (right <= left || bottom <= top) {
                continue;
            }

            candidates.add(new Detection(left, top, right, bottom, confidence, classId));
        }

        Collections.sort(candidates, new Comparator<Detection>() {
            @Override
            public int compare(Detection first, Detection second) {
                return Float.compare(second.getConfidence(), first.getConfidence());
            }
        });

        List<Detection> selected = new ArrayList<>();
        int nmsCandidateCount = Math.min(candidates.size(), DEFAULT_PRE_NMS_TOP_K);
        for (int candidateIndex = 0;
                candidateIndex < nmsCandidateCount
                        && selected.size() < DEFAULT_MAX_DETECTIONS;
                candidateIndex++) {
            Detection candidate = candidates.get(candidateIndex);
            boolean suppressed = false;
            for (Detection accepted : selected) {
                if (intersectionOverUnion(candidate, accepted) > nmsIouThreshold) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) {
                selected.add(candidate);
            }
        }
        return Collections.unmodifiableList(selected);
    }

    private static int validateArguments(
            float[][] prediction,
            int originalWidth,
            int originalHeight,
            float scale,
            float padX,
            float padY,
            float confidenceThreshold,
            float nmsIouThreshold) {
        if (prediction == null || prediction.length != OUTPUT_ROWS) {
            throw new IllegalArgumentException("prediction must contain 6 rows");
        }
        if (prediction[0] == null) {
            throw new IllegalArgumentException("prediction rows must not be null");
        }
        int candidateCount = prediction[0].length;
        for (float[] row : prediction) {
            if (row == null || row.length != candidateCount) {
                throw new IllegalArgumentException("prediction rows must have equal lengths");
            }
        }
        validateGeometry(
                originalWidth,
                originalHeight,
                scale,
                padX,
                padY,
                confidenceThreshold,
                nmsIouThreshold);
        return candidateCount;
    }

    private static void validateGeometry(
            int originalWidth,
            int originalHeight,
            float scale,
            float padX,
            float padY,
            float confidenceThreshold,
            float nmsIouThreshold) {
        if (originalWidth <= 0 || originalHeight <= 0) {
            throw new IllegalArgumentException("original dimensions must be positive");
        }
        if (!isFinite(scale) || scale <= 0f || !isFinite(padX) || !isFinite(padY)) {
            throw new IllegalArgumentException("letterbox parameters must be finite and scale positive");
        }
        if (!isProbability(confidenceThreshold) || !isProbability(nmsIouThreshold)) {
            throw new IllegalArgumentException("thresholds must be finite values from 0 to 1");
        }
    }

    private interface PredictionReader {
        float get(int row, int column);
    }

    private static float intersectionOverUnion(Detection first, Detection second) {
        float intersectionWidth = Math.max(
                0f, Math.min(first.getRight(), second.getRight())
                        - Math.max(first.getLeft(), second.getLeft()));
        float intersectionHeight = Math.max(
                0f, Math.min(first.getBottom(), second.getBottom())
                        - Math.max(first.getTop(), second.getTop()));
        float intersectionArea = intersectionWidth * intersectionHeight;
        float firstArea = (first.getRight() - first.getLeft())
                * (first.getBottom() - first.getTop());
        float secondArea = (second.getRight() - second.getLeft())
                * (second.getBottom() - second.getTop());
        return intersectionArea / (firstArea + secondArea - intersectionArea);
    }

    private static float clip(float value, float minimum, float maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }

    private static boolean isProbability(float value) {
        return isFinite(value) && value >= 0f && value <= 1f;
    }

    private static boolean isFinite(float value) {
        return !Float.isNaN(value) && !Float.isInfinite(value);
    }
}
