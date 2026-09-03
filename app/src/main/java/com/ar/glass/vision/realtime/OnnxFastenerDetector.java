package com.ar.glass.vision.realtime;

import android.content.Context;
import android.graphics.Bitmap;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

import java.io.ByteArrayOutputStream;
import java.io.File;
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
    /** 计算设备描述（NNAPI(GPU/NPU) 或 CPU(线程数)），初始化时确定 */
    private String providerInfo = "未知";
    /** 输出通道数（4+类别数）与候选数；0 表示形状动态、以首次推理实际值为准 */
    private int outputChannels;
    private int outputCandidates;

    /** 当前计算设备描述，供 UI 展示 */
    public synchronized String getProviderInfo() {
        return providerInfo;
    }

    public OnnxFastenerDetector(Context context) {
        this(readAssetBytes(context), false, "模型未打包：" + MODEL_ASSET_NAME);
    }

    /** 从用户选择的模型文件加载（宽松形状校验，更换模型入口使用） */
    public OnnxFastenerDetector(File modelFile) {
        this(readFileBytes(modelFile), true, "模型文件读取失败");
    }

    private OnnxFastenerDetector(byte[] modelBytes, boolean custom, String missingModelError) {
        if (modelBytes == null) {
            inputWorkspace = null;
            initializationError = missingModelError;
            return;
        }

        inputWorkspace = new FastenerInputWorkspace(INPUT_SIZE);

        OrtSession candidateSession = null;
        try {
            environment = OrtEnvironment.getEnvironment();
            // GPU 优先：先尝试 NNAPI 加速，失败自动回退 CPU（兼容性优先）
            boolean nnapiOk = false;
            try {
                try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
                    options.setIntraOpNumThreads(InferenceThreadPolicy.intraOpThreads());
                    options.addNnapi();
                    candidateSession = environment.createSession(modelBytes, options);
                    nnapiOk = true;
                }
            } catch (Throwable gpuErr) {
                closeQuietly(candidateSession);
                candidateSession = null;
            }
            if (!nnapiOk) {
                try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
                    options.setIntraOpNumThreads(InferenceThreadPolicy.intraOpThreads());
                    candidateSession = environment.createSession(modelBytes, options);
                }
            }
            providerInfo = nnapiOk
                    ? "NNAPI(GPU/NPU)"
                    : "CPU(" + InferenceThreadPolicy.intraOpThreads() + "线程)";

            inputName = requireUniqueName(candidateSession.getInputNames(), "input");
            outputName = requireUniqueName(candidateSession.getOutputNames(), "output");
            long[] inputShape = tensorShape(candidateSession.getInputInfo(), inputName);
            long[] outputShape = tensorShape(candidateSession.getOutputInfo(), outputName);
            if (custom) {
                validateCustomModelShapes(inputShape, outputShape);
                outputChannels = (int) outputShape[1];
                outputCandidates = (int) outputShape[2];
            } else {
                validateModelShapes(inputShape, outputShape);
                outputChannels = OUTPUT_CHANNELS;
                outputCandidates = OUTPUT_CANDIDATES;
            }
            session = candidateSession;
        } catch (IllegalArgumentException exception) {
            closeQuietly(candidateSession);
            initializationError = "模型尺寸或输入输出不兼容";
        } catch (OrtException | LinkageError | RuntimeException exception) {
            closeQuietly(candidateSession);
            initializationError = "模型初始化失败";
        }
    }

    private static byte[] readAssetBytes(Context context) {
        if (context == null) {
            return null;
        }
        try {
            return readStream(context.getAssets().open(MODEL_ASSET_NAME));
        } catch (IOException exception) {
            return null;
        }
    }

    private static byte[] readFileBytes(File modelFile) {
        if (modelFile == null || !modelFile.exists() || modelFile.length() <= 0) {
            return null;
        }
        try {
            return readStream(new java.io.FileInputStream(modelFile));
        } catch (IOException exception) {
            return null;
        }
    }

    private static byte[] readStream(InputStream input) throws IOException {
        try (InputStream in = input;
                ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] chunk = new byte[32 * 1024];
            int count;
            while ((count = in.read(chunk)) != -1) {
                output.write(chunk, 0, count);
            }
            return output.toByteArray();
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
        long inferredAtNanos = System.nanoTime();

        List<Detection> detections = YoloPostprocessor.process(
                prediction,
                prediction.length,
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

    /**
     * 自定义模型宽松契约：输入固定 [1,3,640,640]（预处理工作区固定 640），
     * 输出 [1, C, N]（C=4+类别数 ≥5，N>0；N 为动态维度时允许 -1/0）。
     */
    public static void validateCustomModelShapes(long[] inputShape, long[] outputShape) {
        validateShape(inputShape, new long[]{1, 3, INPUT_SIZE, INPUT_SIZE}, "input");
        if (outputShape == null || outputShape.length != 3 || outputShape[0] != 1) {
            throw new IllegalArgumentException("output tensor shape is missing or invalid");
        }
        if (outputShape[1] != -1 && outputShape[1] < 5) {
            throw new IllegalArgumentException("output channel count is incompatible");
        }
        if (outputShape[2] == 0) {
            throw new IllegalArgumentException("output tensor shape is invalid");
        }
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

    private float[][] extractPrediction(
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
        if (prediction.length < 5 || (outputChannels > 0 && prediction.length != outputChannels)) {
            throw new IllegalArgumentException("model output channel count is incompatible");
        }
        for (float[] row : prediction) {
            if (row == null || row.length == 0
                    || (outputCandidates > 0 && row.length != outputCandidates)) {
                throw new IllegalArgumentException("model output candidate count is incompatible");
            }
        }
        if (outputChannels == 0) {
            outputChannels = prediction.length;
            outputCandidates = prediction[0].length;
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
