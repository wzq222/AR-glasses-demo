package com.ar.glass.ui;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.widget.Button;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.ar.glass.R;
import com.ar.glass.vision.realtime.FrameRotation;
import com.ar.glass.vision.realtime.InferenceGate;
import com.ar.glass.vision.realtime.OnnxFastenerDetector;
import com.ar.glass.vision.realtime.Rgba8888Converter;
import com.google.common.util.concurrent.ListenableFuture;

import java.nio.ByteBuffer;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class LiveInspectionActivity extends AppCompatActivity {
    private static final String TAG = "LiveInspection";
    private static final int CAMERA_PERMISSION_REQUEST = 41;
    private static final long INFERENCE_INTERVAL_MILLIS = 500L;

    private final InferenceGate inferenceGate = new InferenceGate(INFERENCE_INTERVAL_MILLIS);

    private PreviewView previewView;
    private DetectionOverlayView overlayView;
    private TextView modelStatusView;
    private TextView metricsView;
    private ExecutorService inferenceExecutor;
    private volatile OnnxFastenerDetector detector;
    private volatile boolean destroyed;
    private boolean cameraRequested;
    private ProcessCameraProvider cameraProvider;
    private ImageAnalysis imageAnalysis;
    private byte[] planeBytes;
    private long previousResultAtMillis;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_live_inspection);

        previewView = findViewById(R.id.live_preview);
        overlayView = findViewById(R.id.detection_overlay);
        modelStatusView = findViewById(R.id.model_status);
        metricsView = findViewById(R.id.inference_metrics);
        Button backButton = findViewById(R.id.live_back);

        previewView.setScaleType(PreviewView.ScaleType.FILL_CENTER);
        backButton.setOnClickListener(view -> finish());

        inferenceExecutor = Executors.newSingleThreadExecutor(runnable -> {
            Thread thread = new Thread(runnable, "fastener-live-inference");
            thread.setDaemon(true);
            return thread;
        });
        initializeDetectorInBackground();

        if (!hasCameraPermission()) {
            ActivityCompat.requestPermissions(
                    this,
                    new String[]{Manifest.permission.CAMERA},
                    CAMERA_PERMISSION_REQUEST);
        }
    }

    @Override
    protected void onStart() {
        super.onStart();
        cameraRequested = true;
        if (hasCameraPermission()) {
            startCamera();
        }
    }

    @Override
    protected void onStop() {
        cameraRequested = false;
        if (imageAnalysis != null) {
            imageAnalysis.clearAnalyzer();
            imageAnalysis = null;
        }
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
        }
        super.onStop();
    }

    @Override
    protected void onDestroy() {
        destroyed = true;
        cameraRequested = false;
        if (imageAnalysis != null) {
            imageAnalysis.clearAnalyzer();
            imageAnalysis = null;
        }
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
            cameraProvider = null;
        }
        if (inferenceExecutor != null) {
            inferenceExecutor.shutdownNow();
        }
        OnnxFastenerDetector currentDetector = detector;
        detector = null;
        if (currentDetector != null) {
            currentDetector.close();
        }
        super.onDestroy();
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode,
            @NonNull String[] permissions,
            @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode != CAMERA_PERMISSION_REQUEST) {
            return;
        }
        if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            if (cameraRequested) {
                startCamera();
            }
        } else {
            metricsView.setText("相机权限未授予，请授权后返回重试");
        }
    }

    private void initializeDetectorInBackground() {
        inferenceExecutor.execute(() -> {
            OnnxFastenerDetector candidate = new OnnxFastenerDetector(getApplicationContext());
            if (destroyed) {
                candidate.close();
                return;
            }
            detector = candidate;
            String status = candidate.isReady()
                    ? "模型状态：已加载"
                    : "模型状态：" + candidate.getInitializationError();
            runOnUiThread(() -> {
                if (!destroyed) {
                    modelStatusView.setText(status);
                    if (!candidate.isReady()) {
                        overlayView.clearDetections();
                    }
                }
            });
        });
    }

    private boolean hasCameraPermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED;
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> providerFuture =
                ProcessCameraProvider.getInstance(this);
        providerFuture.addListener(() -> {
            if (destroyed || !cameraRequested) {
                return;
            }
            try {
                cameraProvider = providerFuture.get();
                bindCameraUseCases();
            } catch (Exception exception) {
                Log.e(TAG, "Unable to start camera", exception);
                metricsView.setText("相机启动失败");
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindCameraUseCases() {
        if (cameraProvider == null || destroyed || !cameraRequested) {
            return;
        }
        cameraProvider.unbindAll();

        Preview preview = new Preview.Builder()
                .setTargetResolution(new Size(640, 480))
                .build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        ImageAnalysis analysis = new ImageAnalysis.Builder()
                .setTargetResolution(new Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build();
        analysis.setAnalyzer(inferenceExecutor, this::analyzeFrame);
        imageAnalysis = analysis;

        cameraProvider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis);
    }

    private void analyzeFrame(ImageProxy image) {
        boolean acquired = false;
        Bitmap inferenceBitmap = null;
        try {
            OnnxFastenerDetector currentDetector = detector;
            if (destroyed || currentDetector == null || !currentDetector.isReady()) {
                return;
            }
            if (!inferenceGate.tryAcquire(SystemClock.elapsedRealtime())) {
                return;
            }
            acquired = true;

            inferenceBitmap = createRotatedBitmap(image);
            OnnxFastenerDetector.DetectionResult result =
                    currentDetector.detect(inferenceBitmap);
            long completedAtMillis = SystemClock.elapsedRealtime();
            double approximateFps = previousResultAtMillis == 0L
                    ? 0.0
                    : 1_000.0 / Math.max(1L, completedAtMillis - previousResultAtMillis);
            previousResultAtMillis = completedAtMillis;
            postResult(result, approximateFps);
        } catch (Exception | LinkageError exception) {
            Log.e(TAG, "Frame inference failed", exception);
            runOnUiThread(() -> {
                if (!destroyed) {
                    modelStatusView.setText("模型状态：已加载（上一帧推理失败）");
                }
            });
        } finally {
            if (inferenceBitmap != null && !inferenceBitmap.isRecycled()) {
                inferenceBitmap.recycle();
            }
            if (acquired) {
                inferenceGate.release();
            }
            image.close();
        }
    }

    private Bitmap createRotatedBitmap(ImageProxy image) {
        ImageProxy.PlaneProxy[] planes = image.getPlanes();
        if (planes.length != 1) {
            throw new IllegalArgumentException("RGBA frame must contain exactly one plane");
        }
        ImageProxy.PlaneProxy plane = planes[0];
        ByteBuffer buffer = plane.getBuffer().duplicate();
        buffer.rewind();
        int byteCount = buffer.remaining();
        if (planeBytes == null || planeBytes.length < byteCount) {
            planeBytes = new byte[byteCount];
        }
        buffer.get(planeBytes, 0, byteCount);

        int[] argb = Rgba8888Converter.toArgb(
                planeBytes,
                image.getWidth(),
                image.getHeight(),
                plane.getRowStride(),
                plane.getPixelStride());
        FrameRotation.RotatedFrame rotated = FrameRotation.rotate(
                argb,
                image.getWidth(),
                image.getHeight(),
                image.getImageInfo().getRotationDegrees());
        Bitmap bitmap = Bitmap.createBitmap(
                rotated.getWidth(), rotated.getHeight(), Bitmap.Config.ARGB_8888);
        bitmap.setPixels(
                rotated.copyPixels(),
                0,
                rotated.getWidth(),
                0,
                0,
                rotated.getWidth(),
                rotated.getHeight());
        return bitmap;
    }

    private void postResult(
            OnnxFastenerDetector.DetectionResult result,
            double approximateFps) {
        String metrics = String.format(
                Locale.CHINA,
                "候选数：%d　推理：%.0f ms　近似FPS：%.1f",
                result.getDetections().size(),
                result.getLatencyMillis(),
                approximateFps);
        runOnUiThread(() -> {
            if (destroyed) {
                return;
            }
            modelStatusView.setText("模型状态：已加载");
            metricsView.setText(metrics);
            overlayView.setDetections(
                    result.getDetections(),
                    result.getOriginalWidth(),
                    result.getOriginalHeight());
        });
    }
}
