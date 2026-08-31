package com.ar.glass.vision.realtime;

import android.content.Context;
import android.content.res.AssetManager;
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
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public final class OnnxFastenerDetector implements Closeable {
    public static final String MODEL_ASSET_NAME = "fastener-target-p2-640.onnx";
    public static final int INPUT_SIZE = 640;
    public static final int OUTPUT_CHANNELS = 6;
    public static final int OUTPUT_CANDIDATES = 34_000;

    private static final long[] INPUT_SHAPE = {1, 3, INPUT_SIZE, INPUT_SIZE};
    private static final int LETTERBOX_COLOR = 0xFF727272;
    private static final int PIXEL_COUNT = INPUT_SIZE * INPUT_SIZE;

    private final FloatBuffer inputBuffer;
    private final int[] pixels;
    private final Bitmap letterboxBitmap;
    private final Canvas letterboxCanvas;
    private final Paint bitmapPaint;
    private final Rect destinationRect;
    private OrtEnvironment environment;
    private OrtSession session;
    private String inputName;
    private String outputName;
    private String initializationError;
    private boolean closed;

    public OnnxFastenerDetector(Context context) {
        if (context == null) {
            inputBuffer = null;
            pixels = null;
            letterboxBitmap = null;
            letterboxCanvas = null;
            bitmapPaint = null;
            destinationRect = null;
            initializationError = "模型初始化失败";
            return;
        }

        inputBuffer = ByteBuffer
                .allocateDirect(PIXEL_COUNT * 3 * Float.BYTES)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        pixels = new int[PIXEL_COUNT];
        letterboxBitmap = Bitmap.createBitmap(
                INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
        letterboxCanvas = new Canvas(letterboxBitmap);
        bitmapPaint = new Paint(Paint.FILTER_BITMAP_FLAG);
        destinationRect = new Rect();

        OrtSession candidateSession = null;
        try {
            environment = OrtEnvironment.getEnvironment();
            byte[] modelBytes = readAsset(context.getAssets(), MODEL_ASSET_NAME);
            try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
                options.setIntraOpNumThreads(InferenceThreadPolicy.intraOpThreads());
                candidateSession = environment.createSession(modelBytes, options);
            }

            inputName = requireUniqueName(candidateSession.getInputNames(), "input");
            outputName = requireUniqueName(candidateSession.getOutputNames(), "output");
            validateModelShapes(
                    tensorShape(candidateSession.getInputInfo(), inputName),
                    tensorShape(candidateSession.getOutputInfo(), outputName));
            session = candidateSession;
        } catch (IOException exception) {
            closeQuietly(candidateSession);
            initializationError = "模型未打包：" + MODEL_ASSET_NAME;
        } catch (IllegalArgumentException exception) {
            closeQuietly(candidateSession);
            initializationError = "模型尺寸或输入输出不兼容";
        } catch (OrtException | LinkageError | RuntimeException exception) {
            closeQuietly(candidateSession);
            initializationError = "模型初始化失败";
        }
    }

    public synchronized boolean isReady() {
        return !closed && session != null;
    }

    public synchronized String getInitializationError() {
        return initializationError;
    }

    public synchronized DetectionResult detect(Bitmap bitmap) throws OrtException {
        if (!isReady()) {
            throw new IllegalStateException(
                    initializationError == null ? "检测器已关闭" : initializationError);
        }
        if (bitmap == null || bitmap.getWidth() <= 0 || bitmap.getHeight() <= 0) {
            throw new IllegalArgumentException("bitmap must have positive dimensions");
        }

        long startedAtNanos = System.nanoTime();
        LetterboxTransform transform = LetterboxTransform.forSquare(
                bitmap.getWidth(), bitmap.getHeight(), INPUT_SIZE);
        fillInputBuffer(bitmap, transform);
        long preprocessedAtNanos = System.nanoTime();

        float[][] prediction;
        try (OnnxTensor inputTensor = OnnxTensor.createTensor(
                environment, inputBuffer, INPUT_SHAPE);
                OrtSession.Result output = session.run(
                        Collections.singletonMap(inputName, inputTensor),
                        Collections.singleton(outputName))) {
            prediction = extractPrediction(output, outputName);
        }
        long inferredAtNanos = System.nanoTime();

        List<Detection> detections = YoloPostprocessor.process(
                prediction,
                bitmap.getWidth(),
                bitmap.getHeight(),
                transform.getScale(),
                transform.getPadX(),
                transform.getPadY());
        long completedAtNanos = System.nanoTime();
        double preprocessMillis = nanosToMillis(preprocessedAtNanos - startedAtNanos);
        double inferenceMillis = nanosToMillis(inferredAtNanos - preprocessedAtNanos);
        double postprocessMillis = nanosToMillis(completedAtNanos - inferredAtNanos);
        double latencyMillis = nanosToMillis(completedAtNanos - startedAtNanos);
        return new DetectionResult(
                detections,
                bitmap.getWidth(),
                bitmap.getHeight(),
                latencyMillis,
                preprocessMillis,
                inferenceMillis,
                postprocessMillis,
                transform);
    }

    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        closeQuietly(session);
        session = null;
        if (letterboxBitmap != null && !letterboxBitmap.isRecycled()) {
            letterboxBitmap.recycle();
        }
    }

    public static void validateModelShapes(long[] inputShape, long[] outputShape) {
        validateShape(inputShape, new long[]{1, 3, INPUT_SIZE, INPUT_SIZE}, "input");
        validateShape(
                outputShape,
                new long[]{1, OUTPUT_CHANNELS, OUTPUT_CANDIDATES},
                "output");
    }

    private static void validateShape(long[] actual, long[] expected, String label) {
        if (actual == null || actual.length != expected.length) {
            throw new IllegalArgumentException(label + " tensor shape is missing or invalid");
        }
        for (int index = 0; index < expected.length; index++) {
            if (actual[index] != expected[index]) {
                throw new IllegalArgumentException(label + " tensor shape is incompatible");
            }
        }
    }

    private static String requireUniqueName(Set<String> names, String label) {
        if (names == null || names.size() != 1) {
            throw new IllegalArgumentException("model must expose exactly one " + label);
        }
        return names.iterator().next();
    }

    private static long[] tensorShape(Map<String, NodeInfo> nodes, String name) {
        NodeInfo node = nodes.get(name);
        if (node == null || !(node.getInfo() instanceof TensorInfo)) {
            throw new IllegalArgumentException("model node is not a tensor");
        }
        return ((TensorInfo) node.getInfo()).getShape();
    }

    private static byte[] readAsset(AssetManager assets, String assetName) throws IOException {
        try (InputStream input = assets.open(assetName);
                ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] chunk = new byte[32 * 1024];
            int count;
            while ((count = input.read(chunk)) != -1) {
                output.write(chunk, 0, count);
            }
            return output.toByteArray();
        }
    }

    private void fillInputBuffer(
            Bitmap source, LetterboxTransform transform) {
        letterboxCanvas.drawColor(LETTERBOX_COLOR);
        destinationRect.set(
                transform.getPadLeft(),
                transform.getPadTop(),
                transform.getPadLeft() + transform.getResizedWidth(),
                transform.getPadTop() + transform.getResizedHeight());
        letterboxCanvas.drawBitmap(source, null, destinationRect, bitmapPaint);

        letterboxBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        inputBuffer.clear();
        for (int index = 0; index < PIXEL_COUNT; index++) {
            int pixel = pixels[index];
            inputBuffer.put(index, ((pixel >> 16) & 0xFF) / 255f);
            inputBuffer.put(PIXEL_COUNT + index, ((pixel >> 8) & 0xFF) / 255f);
            inputBuffer.put(2 * PIXEL_COUNT + index, (pixel & 0xFF) / 255f);
        }
        inputBuffer.position(0);
    }

    private static float[][] extractPrediction(
            OrtSession.Result output, String outputName) throws OrtException {
        Optional<OnnxValue> value = output.get(outputName);
        if (!value.isPresent()) {
            throw new IllegalArgumentException("model output is missing");
        }
        Object raw = value.get().getValue();
        if (!(raw instanceof float[][][])) {
            throw new IllegalArgumentException("model output type is incompatible");
        }
        float[][][] batches = (float[][][]) raw;
        if (batches.length != 1) {
            throw new IllegalArgumentException("model output batch is incompatible");
        }
        float[][] prediction = batches[0];
        if (prediction.length != OUTPUT_CHANNELS) {
            throw new IllegalArgumentException("model output channel count is incompatible");
        }
        for (float[] row : prediction) {
            if (row == null || row.length != OUTPUT_CANDIDATES) {
                throw new IllegalArgumentException("model output candidate count is incompatible");
            }
        }
        return prediction;
    }

    private static void closeQuietly(OrtSession candidate) {
        if (candidate == null) {
            return;
        }
        try {
            candidate.close();
        } catch (OrtException ignored) {
            // Closing is best effort; public errors remain path-free.
        }
    }

    private static double nanosToMillis(long nanos) {
        return nanos / 1_000_000.0;
    }

    public static final class DetectionResult {
        private final List<Detection> detections;
        private final int originalWidth;
        private final int originalHeight;
        private final double latencyMillis;
        private final double preprocessMillis;
        private final double inferenceMillis;
        private final double postprocessMillis;
        private final LetterboxTransform transform;

        DetectionResult(
                List<Detection> detections,
                int originalWidth,
                int originalHeight,
                double latencyMillis,
                double preprocessMillis,
                double inferenceMillis,
                double postprocessMillis,
                LetterboxTransform transform) {
            this.detections = Collections.unmodifiableList(new ArrayList<>(detections));
            this.originalWidth = originalWidth;
            this.originalHeight = originalHeight;
            this.latencyMillis = latencyMillis;
            this.preprocessMillis = preprocessMillis;
            this.inferenceMillis = inferenceMillis;
            this.postprocessMillis = postprocessMillis;
            this.transform = transform;
        }

        public List<Detection> getDetections() { return detections; }
        public int getOriginalWidth() { return originalWidth; }
        public int getOriginalHeight() { return originalHeight; }
        public double getLatencyMillis() { return latencyMillis; }
        public double getPreprocessMillis() { return preprocessMillis; }
        public double getInferenceMillis() { return inferenceMillis; }
        public double getPostprocessMillis() { return postprocessMillis; }
        public LetterboxTransform getTransform() { return transform; }
    }
}
