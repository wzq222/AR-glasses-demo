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
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public final class MarkedPointOnnxVerifier implements Closeable {
    public static final String MODEL_ASSET_NAME = "marked-point-verifier.onnx";
    public static final int INPUT_SIZE = 128;
    public static final float VERIFIER_THRESHOLD = 0.28198338f;
    public static final float PROPOSAL_BYPASS_THRESHOLD = 0.96769285f;
    public static final float NMS_IOU_THRESHOLD = 0.30f;

    private static final int RESIZE_SIZE = 146;
    private static final int CROP_OFFSET = (RESIZE_SIZE - INPUT_SIZE) / 2;
    private static final int PIXEL_COUNT = INPUT_SIZE * INPUT_SIZE;
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};

    private final Bitmap resizeBitmap;
    private final Canvas resizeCanvas;
    private final Paint bitmapPaint;
    private final Rect sourceRect = new Rect();
    private final Rect destinationRect = new Rect(0, 0, RESIZE_SIZE, RESIZE_SIZE);
    private final int[] pixels = new int[PIXEL_COUNT];
    private OrtEnvironment environment;
    private OrtSession session;
    private String inputName;
    private String outputName;
    private String initializationError;
    private FloatBuffer inputBuffer;
    private int inputCapacityCandidates;
    private final boolean useSingleCandidateBatches;
    private boolean closed;

    public MarkedPointOnnxVerifier(Context context) {
        this(context, false);
    }

    public MarkedPointOnnxVerifier(Context context, boolean useNnapi) {
        this(context, useNnapi, false);
    }

    public MarkedPointOnnxVerifier(
            Context context, boolean useNnapi, boolean useXnnpack) {
        if (useNnapi && useXnnpack) {
            throw new IllegalArgumentException("only one verifier accelerator may be enabled");
        }
        useSingleCandidateBatches = useXnnpack;
        if (context == null) {
            resizeBitmap = null;
            resizeCanvas = null;
            bitmapPaint = null;
            initializationError = "防松标记复核模型初始化失败";
            return;
        }
        resizeBitmap = Bitmap.createBitmap(
                RESIZE_SIZE, RESIZE_SIZE, Bitmap.Config.ARGB_8888);
        resizeCanvas = new Canvas(resizeBitmap);
        bitmapPaint = new Paint(Paint.FILTER_BITMAP_FLAG);
        OrtSession candidate = null;
        try {
            environment = OrtEnvironment.getEnvironment();
            byte[] model = readAsset(context.getAssets(), MODEL_ASSET_NAME);
            try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
                if (useXnnpack) {
                    options.setIntraOpNumThreads(1);
                    options.addXnnpack(Collections.singletonMap(
                            "intra_op_num_threads",
                            Integer.toString(InferenceThreadPolicy.intraOpThreads())));
                } else {
                    options.setIntraOpNumThreads(InferenceThreadPolicy.intraOpThreads());
                }
                if (useNnapi) {
                    options.addNnapi();
                }
                candidate = environment.createSession(model, options);
            }
            inputName = requireUniqueName(candidate.getInputNames(), "input");
            outputName = requireUniqueName(candidate.getOutputNames(), "output");
            validateShapes(
                    tensorShape(candidate.getInputInfo(), inputName),
                    tensorShape(candidate.getOutputInfo(), outputName));
            session = candidate;
        } catch (IOException exception) {
            closeQuietly(candidate);
            initializationError = "防松标记复核模型未打包";
        } catch (IllegalArgumentException exception) {
            closeQuietly(candidate);
            initializationError = "防松标记复核模型尺寸不兼容";
        } catch (OrtException | LinkageError | RuntimeException exception) {
            closeQuietly(candidate);
            initializationError = "防松标记复核模型初始化失败";
        }
    }

    public synchronized boolean isReady() {
        return !closed && session != null;
    }

    public synchronized String getInitializationError() {
        return initializationError;
    }

    public synchronized VerificationResult verify(
            Bitmap source, List<Detection> proposals) throws OrtException {
        if (!isReady()) {
            throw new IllegalStateException(initializationError);
        }
        if (source == null || proposals == null) {
            throw new IllegalArgumentException("source and proposals are required");
        }
        if (proposals.isEmpty()) {
            return new VerificationResult(Collections.emptyList(), 0.0, 0.0, 0.0);
        }

        long started = System.nanoTime();
        FloatBuffer values = prepare(source, proposals);
        long preprocessed = System.nanoTime();
        float[] scores = useSingleCandidateBatches
                ? inferSingleCandidateBatches(values, proposals.size())
                : inferDynamicBatch(values, proposals.size());
        long inferred = System.nanoTime();
        List<Detection> detections = filterDetections(proposals, scores);
        long completed = System.nanoTime();
        return new VerificationResult(
                detections,
                nanosToMillis(preprocessed - started),
                nanosToMillis(inferred - preprocessed),
                nanosToMillis(completed - inferred));
    }

    private float[] inferDynamicBatch(FloatBuffer values, int candidates)
            throws OrtException {
        long[] shape = {candidates, 3, INPUT_SIZE, INPUT_SIZE};
        float[][] logits;
        try (OnnxTensor tensor = OnnxTensor.createTensor(environment, values, shape);
                OrtSession.Result result = session.run(
                        Collections.singletonMap(inputName, tensor),
                        Collections.singleton(outputName))) {
            logits = extractLogits(result, outputName, candidates);
        }
        float[] scores = new float[candidates];
        for (int index = 0; index < candidates; index++) {
            scores[index] = softmaxFirst(logits[index]);
        }
        return scores;
    }

    private float[] inferSingleCandidateBatches(FloatBuffer values, int candidates)
            throws OrtException {
        int elements = 3 * PIXEL_COUNT;
        float[] scores = new float[candidates];
        for (int index = 0; index < candidates; index++) {
            FloatBuffer candidate = values.duplicate();
            candidate.position(index * elements);
            candidate.limit((index + 1) * elements);
            candidate = candidate.slice();
            try (OnnxTensor tensor = OnnxTensor.createTensor(
                    environment, candidate, new long[]{1, 3, INPUT_SIZE, INPUT_SIZE});
                    OrtSession.Result result = session.run(
                            Collections.singletonMap(inputName, tensor),
                            Collections.singleton(outputName))) {
                scores[index] = softmaxFirst(
                        extractLogits(result, outputName, 1)[0]);
            }
        }
        return scores;
    }

    static int[] computeSquareCrop(
            Detection detection, int imageWidth, int imageHeight) {
        float width = detection.getRight() - detection.getLeft();
        float height = detection.getBottom() - detection.getTop();
        float centerX = (detection.getLeft() + detection.getRight()) / 2f;
        float centerY = (detection.getTop() + detection.getBottom()) / 2f;
        int side = Math.round(Math.min(
                Math.max(Math.max(width, height) * 1.6f, 64f),
                Math.min(imageWidth, imageHeight)));
        int left = Math.min(
                Math.max(Math.round(centerX - side / 2f), 0), imageWidth - side);
        int top = Math.min(
                Math.max(Math.round(centerY - side / 2f), 0), imageHeight - side);
        return new int[]{left, top, side};
    }

    static List<Detection> filterDetections(
            List<Detection> proposals, float[] verifierScores) {
        if (proposals.size() != verifierScores.length) {
            throw new IllegalArgumentException("verifier score count mismatch");
        }
        List<RankedDetection> eligible = new ArrayList<>();
        for (int index = 0; index < proposals.size(); index++) {
            Detection proposal = proposals.get(index);
            float verifierScore = verifierScores[index];
            if (!Float.isFinite(verifierScore)) {
                throw new IllegalArgumentException("verifier score must be finite");
            }
            if (verifierScore >= VERIFIER_THRESHOLD
                    || proposal.getConfidence() >= PROPOSAL_BYPASS_THRESHOLD) {
                float rank = Math.max(
                        verifierScore / VERIFIER_THRESHOLD,
                        proposal.getConfidence() / PROPOSAL_BYPASS_THRESHOLD);
                eligible.add(new RankedDetection(proposal, verifierScore, rank));
            }
        }
        Collections.sort(eligible, new Comparator<RankedDetection>() {
            @Override
            public int compare(RankedDetection first, RankedDetection second) {
                int rankOrder = Float.compare(second.rank, first.rank);
                if (rankOrder != 0) {
                    return rankOrder;
                }
                int verifierOrder = Float.compare(
                        second.verifierScore, first.verifierScore);
                if (verifierOrder != 0) {
                    return verifierOrder;
                }
                return Float.compare(
                        second.proposal.getConfidence(), first.proposal.getConfidence());
            }
        });
        List<Detection> kept = new ArrayList<>();
        for (RankedDetection candidate : eligible) {
            boolean overlaps = false;
            for (Detection accepted : kept) {
                if (intersectionOverUnion(candidate.proposal, accepted)
                        >= NMS_IOU_THRESHOLD) {
                    overlaps = true;
                    break;
                }
            }
            if (!overlaps) {
                Detection proposal = candidate.proposal;
                kept.add(new Detection(
                        proposal.getLeft(),
                        proposal.getTop(),
                        proposal.getRight(),
                        proposal.getBottom(),
                        Math.max(candidate.verifierScore, proposal.getConfidence()),
                        proposal.getClassId()));
            }
        }
        return Collections.unmodifiableList(kept);
    }

    private FloatBuffer prepare(Bitmap source, List<Detection> proposals) {
        ensureInputCapacity(proposals.size());
        inputBuffer.clear();
        for (int batch = 0; batch < proposals.size(); batch++) {
            int[] crop = computeSquareCrop(
                    proposals.get(batch), source.getWidth(), source.getHeight());
            sourceRect.set(crop[0], crop[1], crop[0] + crop[2], crop[1] + crop[2]);
            resizeCanvas.drawBitmap(source, sourceRect, destinationRect, bitmapPaint);
            resizeBitmap.getPixels(
                    pixels, 0, INPUT_SIZE, CROP_OFFSET, CROP_OFFSET,
                    INPUT_SIZE, INPUT_SIZE);
            int batchBase = batch * 3 * PIXEL_COUNT;
            for (int pixelIndex = 0; pixelIndex < PIXEL_COUNT; pixelIndex++) {
                int pixel = pixels[pixelIndex];
                inputBuffer.put(
                        batchBase + pixelIndex,
                        (((pixel >> 16) & 0xFF) / 255f - MEAN[0]) / STD[0]);
                inputBuffer.put(
                        batchBase + PIXEL_COUNT + pixelIndex,
                        (((pixel >> 8) & 0xFF) / 255f - MEAN[1]) / STD[1]);
                inputBuffer.put(
                        batchBase + 2 * PIXEL_COUNT + pixelIndex,
                        ((pixel & 0xFF) / 255f - MEAN[2]) / STD[2]);
            }
        }
        inputBuffer.position(0);
        inputBuffer.limit(proposals.size() * 3 * PIXEL_COUNT);
        return inputBuffer;
    }

    private void ensureInputCapacity(int candidates) {
        if (inputBuffer != null && candidates <= inputCapacityCandidates) {
            return;
        }
        inputCapacityCandidates = candidates;
        inputBuffer = ByteBuffer
                .allocateDirect(candidates * 3 * PIXEL_COUNT * Float.BYTES)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
    }

    private static float softmaxFirst(float[] logits) {
        if (logits == null || logits.length != 2) {
            throw new IllegalArgumentException("verifier output class count mismatch");
        }
        double maximum = Math.max(logits[0], logits[1]);
        double first = Math.exp(logits[0] - maximum);
        double second = Math.exp(logits[1] - maximum);
        return (float) (first / (first + second));
    }

    private static float intersectionOverUnion(Detection first, Detection second) {
        float left = Math.max(first.getLeft(), second.getLeft());
        float top = Math.max(first.getTop(), second.getTop());
        float right = Math.min(first.getRight(), second.getRight());
        float bottom = Math.min(first.getBottom(), second.getBottom());
        float intersection = Math.max(0f, right - left) * Math.max(0f, bottom - top);
        float firstArea = (first.getRight() - first.getLeft())
                * (first.getBottom() - first.getTop());
        float secondArea = (second.getRight() - second.getLeft())
                * (second.getBottom() - second.getTop());
        float union = firstArea + secondArea - intersection;
        return union <= 0f ? 0f : intersection / union;
    }

    private static void validateShapes(long[] inputShape, long[] outputShape) {
        if (inputShape == null || inputShape.length != 4
                || inputShape[0] != -1 || inputShape[1] != 3
                || inputShape[2] != INPUT_SIZE || inputShape[3] != INPUT_SIZE) {
            throw new IllegalArgumentException("verifier input shape mismatch");
        }
        if (outputShape == null || outputShape.length != 2
                || outputShape[0] != -1 || outputShape[1] != 2) {
            throw new IllegalArgumentException("verifier output shape mismatch");
        }
    }

    private static float[][] extractLogits(
            OrtSession.Result result, String outputName, int expectedBatch)
            throws OrtException {
        Optional<OnnxValue> value = result.get(outputName);
        if (!value.isPresent() || !(value.get().getValue() instanceof float[][])) {
            throw new IllegalArgumentException("verifier output type mismatch");
        }
        float[][] logits = (float[][]) value.get().getValue();
        if (logits.length != expectedBatch) {
            throw new IllegalArgumentException("verifier output batch mismatch");
        }
        return logits;
    }

    private static String requireUniqueName(Set<String> names, String label) {
        if (names == null || names.size() != 1) {
            throw new IllegalArgumentException("verifier must expose one " + label);
        }
        return names.iterator().next();
    }

    private static long[] tensorShape(Map<String, NodeInfo> nodes, String name) {
        NodeInfo node = nodes.get(name);
        if (node == null || !(node.getInfo() instanceof TensorInfo)) {
            throw new IllegalArgumentException("verifier node is not a tensor");
        }
        return ((TensorInfo) node.getInfo()).getShape();
    }

    private static byte[] readAsset(AssetManager assets, String name) throws IOException {
        try (InputStream input = assets.open(name);
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

    private static double nanosToMillis(long nanos) {
        return nanos / 1_000_000.0;
    }

    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        closeQuietly(session);
        session = null;
        if (resizeBitmap != null && !resizeBitmap.isRecycled()) {
            resizeBitmap.recycle();
        }
    }

    private static final class RankedDetection {
        private final Detection proposal;
        private final float verifierScore;
        private final float rank;

        private RankedDetection(
                Detection proposal, float verifierScore, float rank) {
            this.proposal = proposal;
            this.verifierScore = verifierScore;
            this.rank = rank;
        }
    }

    public static final class VerificationResult {
        private final List<Detection> detections;
        private final double preprocessMillis;
        private final double inferenceMillis;
        private final double postprocessMillis;

        private VerificationResult(
                List<Detection> detections,
                double preprocessMillis,
                double inferenceMillis,
                double postprocessMillis) {
            this.detections = detections;
            this.preprocessMillis = preprocessMillis;
            this.inferenceMillis = inferenceMillis;
            this.postprocessMillis = postprocessMillis;
        }

        public List<Detection> getDetections() { return detections; }
        public double getPreprocessMillis() { return preprocessMillis; }
        public double getInferenceMillis() { return inferenceMillis; }
        public double getPostprocessMillis() { return postprocessMillis; }
    }
}
