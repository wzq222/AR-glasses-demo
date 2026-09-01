package com.ar.glass.vision.realtime;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public final class OnnxFastenerDetector implements FastenerDetector {
    public static final String MODEL_ASSET_NAME = "fastener-target-p2-640.onnx";
    public static final int INPUT_SIZE = 640;
    public static final int OUTPUT_CHANNELS = 6;
    public static final int OUTPUT_CANDIDATES = 34_000;

    private static final long[] INPUT_SHAPE = {1, 3, INPUT_SIZE, INPUT_SIZE};
    private final FastenerInputWorkspace inputWorkspace;
    private OrtEnvironment environment;
    private OrtSession session;
    private String inputName;
    private String outputName;
    private String initializationError;
    private boolean closed;

    public OnnxFastenerDetector(Context context) {
        if (context == null) {
            inputWorkspace = null;
            initializationError = "模型初始化失败";
            return;
        }

        inputWorkspace = new FastenerInputWorkspace(INPUT_SIZE);

        OrtSession candidateSession = null;
        try {
            environment = OrtEnvironment.getEnvironment();
            byte[] modelBytes = readAsset(context.getAssets(), MODEL_ASSET_NAME);
            try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
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
        java.nio.FloatBuffer inputBuffer = inputWorkspace.prepare(bitmap, transform);
        long preprocessedAtNanos = System.nanoTime();

        float[][] prediction;
        try (OnnxTensor inputTensor = OnnxTensor.createTensor(
                environment, inputBuffer, INPUT_SHAPE);
                OrtSession.Result output = session.run(
                        Collections.singletonMap(inputName, inputTensor),
                        Collections.singleton(outputName))) {
            prediction = extractPrediction(output, outputName);
        }

        List<Detection> detections = YoloPostprocessor.process(
                prediction,
                bitmap.getWidth(),
                bitmap.getHeight(),
                transform.getScale(),
                transform.getPadX(),
                transform.getPadY());
        double latencyMillis = (System.nanoTime() - startedAtNanos) / 1_000_000.0;
        return new DetectionResult(
                detections,
                bitmap.getWidth(),
                bitmap.getHeight(),
                latencyMillis,
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
        if (inputWorkspace != null) {
            inputWorkspace.close();
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

    public static final class DetectionResult {
        private final List<Detection> detections;
        private final int originalWidth;
        private final int originalHeight;
        private final double latencyMillis;
        private final LetterboxTransform transform;

        DetectionResult(
                List<Detection> detections,
                int originalWidth,
                int originalHeight,
                double latencyMillis,
                LetterboxTransform transform) {
            this.detections = Collections.unmodifiableList(new ArrayList<>(detections));
            this.originalWidth = originalWidth;
            this.originalHeight = originalHeight;
            this.latencyMillis = latencyMillis;
            this.transform = transform;
        }

        public List<Detection> getDetections() { return detections; }
        public int getOriginalWidth() { return originalWidth; }
        public int getOriginalHeight() { return originalHeight; }
        public double getLatencyMillis() { return latencyMillis; }
        public LetterboxTransform getTransform() { return transform; }
    }
}
