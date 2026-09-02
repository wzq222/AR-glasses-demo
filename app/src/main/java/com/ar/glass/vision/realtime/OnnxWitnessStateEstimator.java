package com.ar.glass.vision.realtime;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Rect;

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
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * ROI model that locates the fixed-side and moving-side witness-line segments.
 * Invalid topology or weak evidence fails closed instead of inventing a state.
 */
public final class OnnxWitnessStateEstimator implements Closeable {
    public static final String MODEL_ASSET_NAME = "witness-roi.onnx";
    public static final int INPUT_SIZE = 320;

    private static final String INPUT_NAME = "images";
    private static final String SEGMENTATION_OUTPUT = "segmentation_logits";
    private static final String KEYPOINT_OUTPUT = "keypoint_heatmaps";
    private static final String QUALITY_OUTPUT = "quality_logits";
    private static final Set<String> OUTPUT_NAMES = Collections.unmodifiableSet(
            new LinkedHashSet<>(Arrays.asList(
                    SEGMENTATION_OUTPUT, KEYPOINT_OUTPUT, QUALITY_OUTPUT)));
    private static final int WITNESS_MARK_CHANNEL = 2;
    private static final int MINIMUM_WITNESS_PIXELS = 8;
    private static final float MINIMUM_KEYPOINT_PEAK_PROBABILITY = 0.005f;
    private static final int WITNESS_SUPPORT_RADIUS = 2;
    private static final float MAXIMUM_JOINT_GAP_RATIO = 0.25f;
    private static final float MINIMUM_SEGMENT_LENGTH_PIXELS = 4f;
    private static final int PIXEL_COUNT = INPUT_SIZE * INPUT_SIZE;
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};

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
            initializationError = "防松线状态模型初始化失败";
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
            validateModelContract(candidate);
            session = candidate;
        } catch (IOException exception) {
            closeQuietly(candidate);
            initializationError = "防松线状态模型未打包";
        } catch (IllegalArgumentException exception) {
            closeQuietly(candidate);
            initializationError = "防松线状态模型输入输出不兼容";
        } catch (OrtException | LinkageError | RuntimeException exception) {
            closeQuietly(candidate);
            initializationError = "防松线状态模型初始化失败";
        }
    }

    public synchronized boolean isReady() {
        return !closed && session != null;
    }

    public synchronized String getInitializationError() {
        return initializationError;
    }

    public synchronized WitnessStateEstimate estimate(Bitmap source, WitnessRoi roi)
            throws OrtException {
        if (!isReady()) {
            throw new IllegalStateException(initializationError == null
                    ? "防松线状态模型不可用" : initializationError);
        }
        if (source == null || roi == null
                || roi.getLeft() < 0 || roi.getTop() < 0
                || roi.getRight() > source.getWidth() || roi.getBottom() > source.getHeight()
                || roi.getWidth() <= 0 || roi.getHeight() <= 0) {
            throw new IllegalArgumentException("witness ROI is invalid");
        }
        prepareInput(source, roi);
        long started = System.nanoTime();
        try (OnnxTensor tensor = OnnxTensor.createTensor(
                environment, inputBuffer, new long[]{1, 3, INPUT_SIZE, INPUT_SIZE});
                OrtSession.Result result = session.run(
                        Collections.singletonMap(INPUT_NAME, tensor), OUTPUT_NAMES)) {
            double inferenceMillis = (System.nanoTime() - started) / 1_000_000.0;
            return decodeOutputs(
                    require4d(result, SEGMENTATION_OUTPUT),
                    require4d(result, KEYPOINT_OUTPUT),
                    require2d(result, QUALITY_OUTPUT),
                    inferenceMillis);
        }
    }

    private void prepareInput(Bitmap source, WitnessRoi roi) {
        sourceRect.set(roi.getLeft(), roi.getTop(), roi.getRight(), roi.getBottom());
        inputCanvas.drawBitmap(source, sourceRect, destinationRect, bitmapPaint);
        inputBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        inputBuffer.clear();
        for (int index = 0; index < PIXEL_COUNT; index++) {
            int pixel = pixels[index];
            inputBuffer.put(index, (((pixel >> 16) & 0xFF) / 255f - MEAN[0]) / STD[0]);
            inputBuffer.put(PIXEL_COUNT + index, (((pixel >> 8) & 0xFF) / 255f - MEAN[1]) / STD[1]);
            inputBuffer.put(2 * PIXEL_COUNT + index, ((pixel & 0xFF) / 255f - MEAN[2]) / STD[2]);
        }
        inputBuffer.position(0);
        inputBuffer.limit(3 * PIXEL_COUNT);
    }

    static WitnessStateEstimate decodeOutputs(
            float[][][][] segmentation,
            float[][][][] keypoints,
            float[][] quality,
            double inferenceMillis) {
        validate4d(segmentation, 4, "segmentation");
        validate4d(keypoints, 4, "keypoints");
        if (quality == null || quality.length != 1
                || quality[0] == null || quality[0].length != 4) {
            throw new IllegalArgumentException("quality output shape mismatch");
        }
        for (float value : quality[0]) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException("quality output is nonfinite");
            }
        }

        float[][] witness = segmentation[0][WITNESS_MARK_CHANNEL];
        int witnessPixels = 0;
        for (int y = 0; y < INPUT_SIZE; y++) {
            for (int x = 0; x < INPUT_SIZE; x++) {
                if (witness[y][x] >= 0f) witnessPixels++;
            }
        }
        if (witnessPixels < MINIMUM_WITNESS_PIXELS) {
            throw new IllegalArgumentException("witness mask evidence is empty");
        }

        int[][] points = new int[4][2];
        for (int channel = 0; channel < 4; channel++) {
            float[][] heatmap = keypoints[0][channel];
            float maximum = Float.NEGATIVE_INFINITY;
            int maximumX = -1;
            int maximumY = -1;
            for (int y = 0; y < INPUT_SIZE; y++) {
                for (int x = 0; x < INPUT_SIZE; x++) {
                    if (heatmap[y][x] > maximum) {
                        maximum = heatmap[y][x];
                        maximumX = x;
                        maximumY = y;
                    }
                }
            }
            double exponentialSum = 0.0;
            for (float[] row : heatmap) {
                for (float value : row) exponentialSum += Math.exp(value - maximum);
            }
            double peakProbability = 1.0 / exponentialSum;
            if (!Double.isFinite(peakProbability)
                    || peakProbability < MINIMUM_KEYPOINT_PEAK_PROBABILITY) {
                throw new IllegalArgumentException("keypoint heatmap evidence is not localized");
            }
            if (!hasWitnessSupport(witness, maximumX, maximumY)) {
                throw new IllegalArgumentException("keypoint lacks local witness-mask support");
            }
            points[channel][0] = maximumX;
            points[channel][1] = maximumY;
        }

        double fixedLength = distance(points[0], points[1]);
        double movingLength = distance(points[2], points[3]);
        double jointGap = distance(points[1], points[2]);
        double minimumLength = Math.min(fixedLength, movingLength);
        if (minimumLength < MINIMUM_SEGMENT_LENGTH_PIXELS
                || jointGap > minimumLength * MAXIMUM_JOINT_GAP_RATIO) {
            throw new IllegalArgumentException("decoded witness topology is incoherent");
        }
        double fixedX = points[1][0] - points[0][0];
        double fixedY = points[1][1] - points[0][1];
        double movingX = points[3][0] - points[2][0];
        double movingY = points[3][1] - points[2][1];
        double cosine = Math.abs(
                (fixedX * movingX + fixedY * movingY) / (fixedLength * movingLength));
        double angle = Math.toDegrees(Math.acos(Math.max(-1.0, Math.min(1.0, cosine))));
        if (!Double.isFinite(angle)) {
            throw new IllegalArgumentException("decoded witness angle is nonfinite");
        }
        float[] normalized = new float[8];
        for (int index = 0; index < 4; index++) {
            normalized[index * 2] = points[index][0] / (float) (INPUT_SIZE - 1);
            normalized[index * 2 + 1] = points[index][1] / (float) (INPUT_SIZE - 1);
        }
        return WitnessStateEstimate.measured((float) angle, normalized, inferenceMillis);
    }

    private static boolean hasWitnessSupport(float[][] witness, int pointX, int pointY) {
        int minimumX = Math.max(0, pointX - WITNESS_SUPPORT_RADIUS);
        int maximumX = Math.min(INPUT_SIZE - 1, pointX + WITNESS_SUPPORT_RADIUS);
        int minimumY = Math.max(0, pointY - WITNESS_SUPPORT_RADIUS);
        int maximumY = Math.min(INPUT_SIZE - 1, pointY + WITNESS_SUPPORT_RADIUS);
        for (int y = minimumY; y <= maximumY; y++) {
            for (int x = minimumX; x <= maximumX; x++) {
                if (witness[y][x] >= 0f) return true;
            }
        }
        return false;
    }

    private static double distance(int[] first, int[] second) {
        return Math.hypot(second[0] - first[0], second[1] - first[1]);
    }

    private static void validate4d(float[][][][] values, int channels, String name) {
        if (values == null || values.length != 1 || values[0] == null
                || values[0].length != channels) {
            throw new IllegalArgumentException(name + " output batch shape mismatch");
        }
        for (float[][] channel : values[0]) {
            if (channel == null || channel.length != INPUT_SIZE) {
                throw new IllegalArgumentException(name + " output height mismatch");
            }
            for (float[] row : channel) {
                if (row == null || row.length != INPUT_SIZE) {
                    throw new IllegalArgumentException(name + " output width mismatch");
                }
                for (float value : row) {
                    if (!Float.isFinite(value)) {
                        throw new IllegalArgumentException(name + " output is nonfinite");
                    }
                }
            }
        }
    }

    private static void validateModelContract(OrtSession candidate) throws OrtException {
        if (!candidate.getInputNames().equals(Collections.singleton(INPUT_NAME))
                || !candidate.getOutputNames().equals(OUTPUT_NAMES)) {
            throw new IllegalArgumentException("witness model names mismatch");
        }
        long[] input = tensorShape(candidate.getInputInfo(), INPUT_NAME);
        if (!shapeMatches(input, 3, INPUT_SIZE, INPUT_SIZE)) {
            throw new IllegalArgumentException("witness input shape mismatch");
        }
        if (!shapeMatches(tensorShape(candidate.getOutputInfo(), SEGMENTATION_OUTPUT), 4, INPUT_SIZE, INPUT_SIZE)
                || !shapeMatches(tensorShape(candidate.getOutputInfo(), KEYPOINT_OUTPUT), 4, INPUT_SIZE, INPUT_SIZE)
                || !shapeMatches(tensorShape(candidate.getOutputInfo(), QUALITY_OUTPUT), 4)) {
            throw new IllegalArgumentException("witness output shape mismatch");
        }
    }

    private static boolean shapeMatches(long[] shape, long... afterBatch) {
        if (shape == null || shape.length != afterBatch.length + 1
                || (shape[0] != -1 && shape[0] <= 0)) return false;
        for (int index = 0; index < afterBatch.length; index++) {
            long actual = shape[index + 1];
            if (actual != -1 && actual != afterBatch[index]) return false;
        }
        return true;
    }

    private static long[] tensorShape(Map<String, NodeInfo> nodes, String name) {
        NodeInfo node = nodes.get(name);
        if (node == null || !(node.getInfo() instanceof TensorInfo)) {
            throw new IllegalArgumentException("witness node is not a tensor");
        }
        return ((TensorInfo) node.getInfo()).getShape();
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

    private static byte[] readAsset(Context context, String name) throws IOException {
        try (InputStream input = context.getAssets().open(name);
                ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] chunk = new byte[32 * 1024];
            int count;
            while ((count = input.read(chunk)) != -1) output.write(chunk, 0, count);
            return output.toByteArray();
        }
    }

    private static void closeQuietly(OrtSession candidate) {
        if (candidate == null) return;
        try {
            candidate.close();
        } catch (OrtException ignored) {
            // Best effort.
        }
    }

    @Override public synchronized void close() {
        if (closed) return;
        closed = true;
        closeQuietly(session);
        session = null;
        if (inputBitmap != null && !inputBitmap.isRecycled()) inputBitmap.recycle();
    }
}
