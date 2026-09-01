package com.ar.glass.vision.realtime;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Rect;

import com.ar.glass.vision.fastener.WitnessStateEstimate;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public final class OnnxWitnessStateEstimator implements Closeable {
    public static final String MODEL_ASSET_NAME = "witness-roi.onnx";
    public static final int INPUT_SIZE = 320;
    /** Experimental evidence gates; these are refusal thresholds, not accuracy claims. */
    public static final int EXPERIMENTAL_WITNESS_MARK_CHANNEL = 2;
    public static final float EXPERIMENTAL_SEGMENTATION_LOGIT_THRESHOLD = 0f;
    public static final int EXPERIMENTAL_MINIMUM_WITNESS_PIXELS = 8;
    public static final int EXPERIMENTAL_QUALITY_MARK_INTEGRITY_CHANNEL = 0;
    public static final int EXPERIMENTAL_QUALITY_OCCLUSION_CHANNEL = 1;
    public static final int EXPERIMENTAL_QUALITY_BLUR_CHANNEL = 2;
    public static final int EXPERIMENTAL_QUALITY_TOPOLOGY_CHANNEL = 3;
    public static final float EXPERIMENTAL_QUALITY_PROBABILITY_THRESHOLD = 0.5f;
    public static final float EXPERIMENTAL_MINIMUM_KEYPOINT_DYNAMIC_RANGE = 1.0e-3f;

    private static final int PIXEL_COUNT = INPUT_SIZE * INPUT_SIZE;
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};
    private static final String INPUT_NAME = "images";
    private static final String SEGMENTATION_OUTPUT = "segmentation_logits";
    private static final String KEYPOINT_OUTPUT = "keypoint_heatmaps";
    private static final String QUALITY_OUTPUT = "quality_logits";
    private static final Set<String> OUTPUT_NAMES = Collections.unmodifiableSet(
            new LinkedHashSet<>(Arrays.asList(
                    SEGMENTATION_OUTPUT, KEYPOINT_OUTPUT, QUALITY_OUTPUT)));

    private final Bitmap inputBitmap;
    private final Canvas inputCanvas;
    private final Paint bitmapPaint;
    private final Rect sourceRect = new Rect();
    private final Rect destinationRect = new Rect(0, 0, INPUT_SIZE, INPUT_SIZE);
    private final int[] pixels = new int[PIXEL_COUNT];
    private final FloatBuffer inputBuffer = ByteBuffer
            .allocateDirect(3 * PIXEL_COUNT * Float.BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer();
    private OrtEnvironment environment;
    private OrtSession session;
    private String initializationError;
    private boolean closed;

    public OnnxWitnessStateEstimator(Context context) {
        if (context == null) {
            inputBitmap = null;
            inputCanvas = null;
            bitmapPaint = null;
            initializationError = "实验状态模型初始化失败";
            return;
        }
        inputBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
        inputCanvas = new Canvas(inputBitmap);
        bitmapPaint = new Paint(Paint.FILTER_BITMAP_FLAG);
        OrtSession candidate = null;
        try {
            environment = OrtEnvironment.getEnvironment();
            byte[] model = readAsset(context, MODEL_ASSET_NAME);
            try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
                options.setIntraOpNumThreads(InferenceThreadPolicy.intraOpThreads());
                candidate = environment.createSession(model, options);
            }
            if (!candidate.getInputNames().equals(Collections.singleton(INPUT_NAME))) {
                throw new IllegalArgumentException("witness input name mismatch");
            }
            Map<String, long[]> outputShapes = new LinkedHashMap<>();
            for (String output : OUTPUT_NAMES) {
                outputShapes.put(output, tensorShape(candidate.getOutputInfo(), output));
            }
            if (!candidate.getOutputNames().equals(OUTPUT_NAMES)) {
                throw new IllegalArgumentException("witness output names mismatch");
            }
            validateModelShapes(
                    tensorShape(candidate.getInputInfo(), INPUT_NAME), outputShapes);
            session = candidate;
        } catch (IOException exception) {
            closeQuietly(candidate);
            initializationError = "实验状态模型未打包";
        } catch (IllegalArgumentException exception) {
            closeQuietly(candidate);
            initializationError = "实验状态模型尺寸不兼容";
        } catch (OrtException | LinkageError | RuntimeException exception) {
            closeQuietly(candidate);
            initializationError = "实验状态模型初始化失败";
        }
    }

    public synchronized boolean isReady() {
        return !closed && session != null;
    }

    public synchronized String getInitializationError() {
        return initializationError;
    }

    public synchronized WitnessStateEstimate estimate(Bitmap source, SquareRoi roi)
            throws OrtException {
        if (!isReady()) {
            throw new IllegalStateException(initializationError);
        }
        if (source == null || roi == null
                || roi.getLeft() < 0 || roi.getTop() < 0
                || roi.getRight() > source.getWidth()
                || roi.getBottom() > source.getHeight()
                || roi.getSide() <= 0) {
            throw new IllegalArgumentException("witness ROI is invalid");
        }
        prepareInput(source, roi);
        long started = System.nanoTime();
        try (OnnxTensor tensor = OnnxTensor.createTensor(
                environment, inputBuffer, new long[]{1, 3, INPUT_SIZE, INPUT_SIZE});
                OrtSession.Result result = session.run(
                        Collections.singletonMap(INPUT_NAME, tensor), OUTPUT_NAMES)) {
            long completed = System.nanoTime();
            float[][][][] segmentation = require4d(result, SEGMENTATION_OUTPUT);
            float[][][][] keypoints = require4d(result, KEYPOINT_OUTPUT);
            float[][] quality = require2d(result, QUALITY_OUTPUT);
            validateDecodedOutputs(segmentation, keypoints, quality);
            return decodeEstimate(
                    keypoints[0], (completed - started) / 1_000_000.0);
        }
    }

    private void prepareInput(Bitmap source, SquareRoi roi) {
        sourceRect.set(roi.getLeft(), roi.getTop(), roi.getRight(), roi.getBottom());
        inputCanvas.drawBitmap(source, sourceRect, destinationRect, bitmapPaint);
        inputBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        inputBuffer.clear();
        for (int index = 0; index < PIXEL_COUNT; index++) {
            int pixel = pixels[index];
            inputBuffer.put(index,
                    (((pixel >> 16) & 0xFF) / 255f - MEAN[0]) / STD[0]);
            inputBuffer.put(PIXEL_COUNT + index,
                    (((pixel >> 8) & 0xFF) / 255f - MEAN[1]) / STD[1]);
            inputBuffer.put(2 * PIXEL_COUNT + index,
                    ((pixel & 0xFF) / 255f - MEAN[2]) / STD[2]);
        }
        inputBuffer.position(0);
        inputBuffer.limit(3 * PIXEL_COUNT);
    }

    static void validateModelShapes(long[] inputShape, Map<String, long[]> outputShapes) {
        if (!shapeMatches(inputShape, 3, INPUT_SIZE, INPUT_SIZE)
                || outputShapes == null
                || outputShapes.size() != 3
                || !shapeMatches(outputShapes.get(SEGMENTATION_OUTPUT), 4, INPUT_SIZE, INPUT_SIZE)
                || !shapeMatches(outputShapes.get(KEYPOINT_OUTPUT), 4, INPUT_SIZE, INPUT_SIZE)
                || !shapeMatches(outputShapes.get(QUALITY_OUTPUT), 4)
                || outputShapes.get(SEGMENTATION_OUTPUT)[0] != inputShape[0]
                || outputShapes.get(KEYPOINT_OUTPUT)[0] != inputShape[0]
                || outputShapes.get(QUALITY_OUTPUT)[0] != inputShape[0]) {
            throw new IllegalArgumentException("witness model tensor shape mismatch");
        }
    }

    static WitnessStateEstimate decodeEstimate(
            float[][][] keypointHeatmaps, double inferenceMillis) {
        if (keypointHeatmaps == null || keypointHeatmaps.length != 4) {
            throw new IllegalArgumentException("keypoint heatmap channel mismatch");
        }
        int[][] points = new int[4][2];
        for (int channel = 0; channel < 4; channel++) {
            float[][] heatmap = keypointHeatmaps[channel];
            if (heatmap == null || heatmap.length != INPUT_SIZE) {
                throw new IllegalArgumentException("keypoint heatmap height mismatch");
            }
            float maximum = -Float.MAX_VALUE;
            int maximumX = -1;
            int maximumY = -1;
            for (int y = 0; y < INPUT_SIZE; y++) {
                if (heatmap[y] == null || heatmap[y].length != INPUT_SIZE) {
                    throw new IllegalArgumentException("keypoint heatmap width mismatch");
                }
                for (int x = 0; x < INPUT_SIZE; x++) {
                    float value = heatmap[y][x];
                    if (!Float.isFinite(value)) {
                        throw new IllegalArgumentException("keypoint heatmap is nonfinite");
                    }
                    if (value > maximum) {
                        maximum = value;
                        maximumX = x;
                        maximumY = y;
                    }
                }
            }
            points[channel][0] = maximumX;
            points[channel][1] = maximumY;
        }
        double fixedX = points[1][0] - points[0][0];
        double fixedY = points[1][1] - points[0][1];
        double movingX = points[3][0] - points[2][0];
        double movingY = points[3][1] - points[2][1];
        double fixedLength = Math.hypot(fixedX, fixedY);
        double movingLength = Math.hypot(movingX, movingY);
        if (fixedLength <= 1.0e-6 || movingLength <= 1.0e-6) {
            throw new IllegalArgumentException("decoded witness segment is degenerate");
        }
        double cosine = Math.abs(
                (fixedX * movingX + fixedY * movingY) / (fixedLength * movingLength));
        cosine = Math.max(-1.0, Math.min(1.0, cosine));
        double angle = Math.toDegrees(Math.acos(cosine));
        if (!Double.isFinite(angle)) {
            throw new IllegalArgumentException("decoded witness angle is nonfinite");
        }
        return WitnessStateEstimate.experimental((float) angle, inferenceMillis);
    }

    static void validateDecodedOutputs(
            float[][][][] segmentation,
            float[][][][] keypoints,
            float[][] quality) {
        validate4dFinite(segmentation, 4, INPUT_SIZE, INPUT_SIZE);
        validate4dFinite(keypoints, 4, INPUT_SIZE, INPUT_SIZE);
        if (quality == null || quality.length != 1
                || quality[0] == null || quality[0].length != 4) {
            throw new IllegalArgumentException("quality output shape mismatch");
        }
        for (float value : quality[0]) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException("quality output is nonfinite");
            }
        }
        validateSemanticEvidence(segmentation[0], keypoints[0], quality[0]);
    }

    private static void validateSemanticEvidence(
            float[][][] segmentation,
            float[][][] keypoints,
            float[] quality) {
        int witnessPixels = 0;
        float[][] witness = segmentation[EXPERIMENTAL_WITNESS_MARK_CHANNEL];
        for (int y = 0; y < INPUT_SIZE; y++) {
            for (int x = 0; x < INPUT_SIZE; x++) {
                if (witness[y][x] >= EXPERIMENTAL_SEGMENTATION_LOGIT_THRESHOLD) {
                    witnessPixels++;
                    if (witnessPixels >= EXPERIMENTAL_MINIMUM_WITNESS_PIXELS) {
                        break;
                    }
                }
            }
            if (witnessPixels >= EXPERIMENTAL_MINIMUM_WITNESS_PIXELS) {
                break;
            }
        }
        if (witnessPixels < EXPERIMENTAL_MINIMUM_WITNESS_PIXELS) {
            throw new IllegalArgumentException("witness mask evidence is empty");
        }

        float markIntegrity = sigmoid(
                quality[EXPERIMENTAL_QUALITY_MARK_INTEGRITY_CHANNEL]);
        float occlusion = sigmoid(quality[EXPERIMENTAL_QUALITY_OCCLUSION_CHANNEL]);
        float blur = sigmoid(quality[EXPERIMENTAL_QUALITY_BLUR_CHANNEL]);
        float topology = sigmoid(quality[EXPERIMENTAL_QUALITY_TOPOLOGY_CHANNEL]);
        if (markIntegrity < EXPERIMENTAL_QUALITY_PROBABILITY_THRESHOLD
                || topology < EXPERIMENTAL_QUALITY_PROBABILITY_THRESHOLD
                || occlusion > EXPERIMENTAL_QUALITY_PROBABILITY_THRESHOLD
                || blur > EXPERIMENTAL_QUALITY_PROBABILITY_THRESHOLD) {
            throw new IllegalArgumentException("witness quality evidence failed");
        }

        for (float[][] heatmap : keypoints) {
            float minimum = Float.POSITIVE_INFINITY;
            float maximum = Float.NEGATIVE_INFINITY;
            for (float[] row : heatmap) {
                for (float value : row) {
                    minimum = Math.min(minimum, value);
                    maximum = Math.max(maximum, value);
                }
            }
            if (maximum - minimum < EXPERIMENTAL_MINIMUM_KEYPOINT_DYNAMIC_RANGE) {
                throw new IllegalArgumentException("keypoint heatmap evidence is flat");
            }
        }
    }

    private static float sigmoid(float value) {
        if (value >= 0f) {
            double exponential = Math.exp(-value);
            return (float) (1.0 / (1.0 + exponential));
        }
        double exponential = Math.exp(value);
        return (float) (exponential / (1.0 + exponential));
    }

    private static void validate4dFinite(
            float[][][][] values, int channels, int height, int width) {
        if (values == null || values.length != 1 || values[0] == null
                || values[0].length != channels) {
            throw new IllegalArgumentException("witness output batch shape mismatch");
        }
        for (float[][] channel : values[0]) {
            if (channel == null || channel.length != height) {
                throw new IllegalArgumentException("witness output height mismatch");
            }
            for (float[] row : channel) {
                if (row == null || row.length != width) {
                    throw new IllegalArgumentException("witness output width mismatch");
                }
                for (float value : row) {
                    if (!Float.isFinite(value)) {
                        throw new IllegalArgumentException("witness output is nonfinite");
                    }
                }
            }
        }
    }

    private static boolean shapeMatches(long[] shape, long... dimensionsAfterBatch) {
        if (shape == null || shape.length != dimensionsAfterBatch.length + 1
                || (shape[0] != -1 && shape[0] <= 0)) {
            return false;
        }
        for (int index = 0; index < dimensionsAfterBatch.length; index++) {
            if (shape[index + 1] != dimensionsAfterBatch[index]) {
                return false;
            }
        }
        return true;
    }

    private static float[][][][] require4d(OrtSession.Result result, String name)
            throws OrtException {
        Optional<OnnxValue> value = result.get(name);
        if (!value.isPresent() || !(value.get().getValue() instanceof float[][][][])) {
            throw new IllegalArgumentException(name + " output type mismatch");
        }
        return (float[][][][]) value.get().getValue();
    }

    private static float[][] require2d(OrtSession.Result result, String name)
            throws OrtException {
        Optional<OnnxValue> value = result.get(name);
        if (!value.isPresent() || !(value.get().getValue() instanceof float[][])) {
            throw new IllegalArgumentException(name + " output type mismatch");
        }
        return (float[][]) value.get().getValue();
    }

    private static long[] tensorShape(Map<String, NodeInfo> nodes, String name) {
        NodeInfo node = nodes.get(name);
        if (node == null || !(node.getInfo() instanceof TensorInfo)) {
            throw new IllegalArgumentException("witness node is not a tensor");
        }
        return ((TensorInfo) node.getInfo()).getShape();
    }

    private static byte[] readAsset(Context context, String name) throws IOException {
        try (InputStream input = context.getAssets().open(name);
                ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] chunk = new byte[32 * 1024];
            int count;
            while ((count = input.read(chunk)) != -1) {
                output.write(chunk, 0, count);
            }
            return output.toByteArray();
        }
    }

    private static void closeQuietly(OrtSession candidate) {
        if (candidate == null) {
            return;
        }
        try {
            candidate.close();
        } catch (OrtException ignored) {
            // Best effort.
        }
    }

    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        closeQuietly(session);
        session = null;
        if (inputBitmap != null && !inputBitmap.isRecycled()) {
            inputBitmap.recycle();
        }
    }
}
