package com.ar.glass.vision.realtime;

import android.content.Context;
import android.graphics.Bitmap;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.List;

public final class NcnnFastenerDetector implements FastenerDetector {
    public static final String PARAM_ASSET_NAME = "model.ncnn.param";
    public static final String BIN_ASSET_NAME = "model.ncnn.bin";
    public static final String INPUT_BLOB_NAME = "in0";
    public static final String OUTPUT_BLOB_NAME = "out0";
    public static final int INPUT_SIZE = 640;
    public static final int OUTPUT_CHANNELS = 6;
    public static final int OUTPUT_CANDIDATES = 34_000;

    private static boolean nativeLoadAttempted;
    private static boolean nativeLoaded;

    private final FastenerInputWorkspace inputWorkspace;
    private final FloatBuffer outputBuffer;
    private final float confidenceThreshold;
    private long nativeHandle;
    private String initializationError;
    private boolean closed;

    public NcnnFastenerDetector(Context context) {
        this(context, YoloPostprocessor.DEFAULT_CONFIDENCE_THRESHOLD);
    }

    public NcnnFastenerDetector(Context context, float confidenceThreshold) {
        if (Float.isNaN(confidenceThreshold) || Float.isInfinite(confidenceThreshold)
                || confidenceThreshold < 0f || confidenceThreshold > 1f) {
            throw new IllegalArgumentException("confidence threshold must be from 0 to 1");
        }
        this.confidenceThreshold = confidenceThreshold;
        if (context == null) {
            inputWorkspace = null;
            outputBuffer = null;
            initializationError = "模型初始化失败";
            return;
        }
        inputWorkspace = new FastenerInputWorkspace(INPUT_SIZE);
        outputBuffer = ByteBuffer
                .allocateDirect(OUTPUT_CHANNELS * OUTPUT_CANDIDATES * Float.BYTES)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        if (!ensureNativeLoaded()) {
            initializationError = "ncnn 运行库未打包";
            return;
        }
        try {
            nativeHandle = nativeCreate(
                    context.getAssets(),
                    PARAM_ASSET_NAME,
                    BIN_ASSET_NAME,
                    INPUT_BLOB_NAME,
                    OUTPUT_BLOB_NAME,
                    InferenceThreadPolicy.intraOpThreads());
            if (nativeHandle == 0L) {
                initializationError = "ncnn 模型未打包或不兼容";
            }
        } catch (LinkageError | RuntimeException exception) {
            initializationError = "ncnn 模型初始化失败";
            nativeHandle = 0L;
        }
    }

    @Override
    public synchronized boolean isReady() {
        return !closed && nativeHandle != 0L;
    }

    @Override
    public synchronized String getInitializationError() {
        return initializationError;
    }

    @Override
    public synchronized OnnxFastenerDetector.DetectionResult detect(Bitmap bitmap) {
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
        FloatBuffer inputBuffer = inputWorkspace.prepare(bitmap, transform);
        long preprocessedAtNanos = System.nanoTime();

        outputBuffer.clear();
        if (!nativeInfer(nativeHandle, inputBuffer, outputBuffer)) {
            throw new IllegalStateException("ncnn 推理输出不兼容");
        }
        outputBuffer.position(0);
        long inferredAtNanos = System.nanoTime();

        List<Detection> detections = YoloPostprocessor.process(
                outputBuffer,
                OUTPUT_CANDIDATES,
                bitmap.getWidth(),
                bitmap.getHeight(),
                transform.getScale(),
                transform.getPadX(),
                transform.getPadY(),
                confidenceThreshold,
                YoloPostprocessor.DEFAULT_NMS_IOU_THRESHOLD);
        long completedAtNanos = System.nanoTime();
        return new OnnxFastenerDetector.DetectionResult(
                detections,
                bitmap.getWidth(),
                bitmap.getHeight(),
                nanosToMillis(completedAtNanos - startedAtNanos),
                nanosToMillis(preprocessedAtNanos - startedAtNanos),
                nanosToMillis(inferredAtNanos - preprocessedAtNanos),
                nanosToMillis(completedAtNanos - inferredAtNanos),
                transform);
    }

    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        if (nativeHandle != 0L) {
            nativeDestroy(nativeHandle);
            nativeHandle = 0L;
        }
        if (inputWorkspace != null) {
            inputWorkspace.close();
        }
    }

    private static synchronized boolean ensureNativeLoaded() {
        if (!nativeLoadAttempted) {
            nativeLoadAttempted = true;
            try {
                System.loadLibrary("crrc_ncnn");
                nativeLoaded = true;
            } catch (LinkageError error) {
                nativeLoaded = false;
            }
        }
        return nativeLoaded;
    }

    private static double nanosToMillis(long nanos) {
        return nanos / 1_000_000.0;
    }

    private static native long nativeCreate(
            android.content.res.AssetManager assets,
            String paramAssetName,
            String binAssetName,
            String inputBlobName,
            String outputBlobName,
            int threads);

    private static native boolean nativeInfer(
            long handle, FloatBuffer input, FloatBuffer output);

    private static native void nativeDestroy(long handle);
}
