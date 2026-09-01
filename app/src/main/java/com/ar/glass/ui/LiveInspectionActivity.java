package com.ar.glass.ui;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.view.Gravity;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.UseCaseGroup;
import androidx.camera.core.ViewPort;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.ar.glass.R;
import com.ar.glass.BuildConfig;
import com.ar.glass.vision.realtime.DetectorFactory;
import com.ar.glass.vision.realtime.FastenerDetector;
import com.ar.glass.vision.realtime.FrameCropper;
import com.ar.glass.vision.realtime.FrameRotation;
import com.ar.glass.vision.realtime.InferenceGate;
import com.ar.glass.vision.realtime.OnnxWitnessStateEstimator;
import com.ar.glass.vision.realtime.OnnxFastenerDetector;
import com.ar.glass.vision.realtime.Rgba8888Converter;
import com.ar.glass.vision.realtime.Detection;
import com.ar.glass.vision.realtime.SquareRoi;
import com.ar.glass.vision.fastener.WitnessReviewHint;
import com.ar.glass.vision.fastener.WitnessStateEstimate;
import com.google.common.util.concurrent.ListenableFuture;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class LiveInspectionActivity extends AppCompatActivity {
    private static final String TAG = "LiveInspection";
    private static final int CAMERA_PERMISSION_REQUEST = 41;
    private static final long INFERENCE_COOLDOWN_MILLIS = 1_000L;

    private final InferenceGate inferenceGate = new InferenceGate(INFERENCE_COOLDOWN_MILLIS);

    private PreviewView previewView;
    private DetectionOverlayView overlayView;
    private TextView modelStatusView;
    private TextView metricsView;
    private ExecutorService inferenceExecutor;
    private volatile FastenerDetector detector;
    private volatile OnnxWitnessStateEstimator stateEstimator;
    private volatile boolean destroyed;
    private volatile boolean inferenceFailed;
    private volatile boolean reviewInProgress;
    private boolean cameraRequested;
    private ProcessCameraProvider cameraProvider;
    private ImageAnalysis imageAnalysis;
    private byte[] planeBytes;
    private int[] sourcePixels;
    private int[] croppedPixels;
    private int[] rotatedPixels;
    private final List<Bitmap> frameBitmaps = new ArrayList<>();
    private Bitmap frameBitmap;
    private Bitmap detectionFrame;
    private AlertDialog activeReviewDialog;
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
        overlayView.setOnDetectionTapListener(this::onDetectionTapped);
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
        inferenceFailed = false;
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
        AlertDialog reviewDialog = activeReviewDialog;
        activeReviewDialog = null;
        if (reviewDialog != null) {
            reviewDialog.dismiss();
        }
        if (imageAnalysis != null) {
            imageAnalysis.clearAnalyzer();
            imageAnalysis = null;
        }
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
            cameraProvider = null;
        }
        ExecutorService executor = inferenceExecutor;
        if (executor != null) {
            executor.execute(this::closeInferenceResources);
            executor.shutdown();
        }
        Bitmap currentDetectionFrame = detectionFrame;
        detectionFrame = null;
        if (currentDetectionFrame != null && !currentDetectionFrame.isRecycled()) {
            currentDetectionFrame.recycle();
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
            metricsView.setText(R.string.live_camera_permission_denied);
        }
    }

    private void initializeDetectorInBackground() {
        inferenceExecutor.execute(() -> {
            FastenerDetector candidate = DetectorFactory.create(
                    getApplicationContext(),
                    BuildConfig.DETECTOR_BACKEND,
                    BuildConfig.NCNN_CONFIDENCE_THRESHOLD,
                    BuildConfig.NCNN_VULKAN,
                    BuildConfig.NCNN_VULKAN_FP16,
                    BuildConfig.MARKED_POINT_VERIFIER_ENABLED,
                    BuildConfig.MARKED_POINT_VERIFIER_NNAPI,
                    BuildConfig.MARKED_POINT_VERIFIER_XNNPACK);
            if (destroyed) {
                candidate.close();
                return;
            }
            detector = candidate;
            OnnxWitnessStateEstimator stateCandidate =
                    BuildConfig.WITNESS_STATE_EXPERIMENTAL_ENABLED
                            ? new OnnxWitnessStateEstimator(getApplicationContext())
                            : null;
            if (destroyed) {
                if (stateCandidate != null) {
                    stateCandidate.close();
                }
                candidate.close();
                detector = null;
                return;
            }
            stateEstimator = stateCandidate;
            String initializationError = candidate.getInitializationError();
            int statusResource;
            if (candidate.isReady()) {
                statusResource = R.string.live_model_ready;
            } else if (("模型未打包：" + OnnxFastenerDetector.MODEL_ASSET_NAME)
                    .equals(initializationError)) {
                statusResource = R.string.live_model_missing;
            } else {
                statusResource = R.string.live_model_initialization_error;
            }
            if (!candidate.isReady()) {
                Log.w(TAG, "Detector initialization failed: "
                        + candidate.getInitializationError());
            }
            runOnUiThread(() -> {
                if (!destroyed) {
                    modelStatusView.setText(statusResource);
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
        providerFuture.addListener(
                () -> previewView.post(() -> {
                    if (destroyed || !cameraRequested) {
                        return;
                    }
                    try {
                        cameraProvider = providerFuture.get();
                        bindCameraUseCases();
                    } catch (Exception exception) {
                        Log.e(TAG, "Unable to start camera", exception);
                        metricsView.setText(R.string.live_camera_start_failed);
                    }
                }),
                ContextCompat.getMainExecutor(this));
    }

    private void bindCameraUseCases() {
        if (cameraProvider == null || destroyed || !cameraRequested) {
            return;
        }
        ViewPort viewPort = previewView.getViewPort();
        if (viewPort == null) {
            Log.e(TAG, "Preview viewport is unavailable");
            metricsView.setText(R.string.live_camera_start_failed);
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

        UseCaseGroup useCaseGroup = new UseCaseGroup.Builder()
                .setViewPort(viewPort)
                .addUseCase(preview)
                .addUseCase(analysis)
                .build();
        cameraProvider.bindToLifecycle(
                this, CameraSelector.DEFAULT_BACK_CAMERA, useCaseGroup);
    }

    private void analyzeFrame(ImageProxy image) {
        boolean acquired = false;
        try {
            FastenerDetector currentDetector = detector;
            if (destroyed || inferenceFailed || reviewInProgress
                    || currentDetector == null || !currentDetector.isReady()) {
                return;
            }
            if (!inferenceGate.tryAcquire(SystemClock.elapsedRealtime())) {
                return;
            }
            acquired = true;

            Bitmap inferenceBitmap = createRotatedBitmap(image);
            OnnxFastenerDetector.DetectionResult result =
                    currentDetector.detect(inferenceBitmap);
            long completedAtMillis = SystemClock.elapsedRealtime();
            double approximateFps = previousResultAtMillis == 0L
                    ? 0.0
                    : 1_000.0 / Math.max(1L, completedAtMillis - previousResultAtMillis);
            previousResultAtMillis = completedAtMillis;
            Bitmap frozenDetectionFrame = inferenceBitmap.copy(
                    Bitmap.Config.ARGB_8888, false);
            if (frozenDetectionFrame == null) {
                throw new IllegalStateException("unable to freeze detection frame");
            }
            if (reviewInProgress) {
                frozenDetectionFrame.recycle();
                return;
            }
            postResult(result, approximateFps, frozenDetectionFrame);
        } catch (Exception | LinkageError exception) {
            inferenceFailed = true;
            Log.e(TAG, "Frame inference failed", exception);
            runOnUiThread(() -> {
                if (!destroyed) {
                    overlayView.clearDetections();
                    modelStatusView.setText(R.string.live_frame_inference_failed);
                }
            });
        } finally {
            if (acquired) {
                inferenceGate.release(SystemClock.elapsedRealtime());
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

        int width = image.getWidth();
        int height = image.getHeight();
        int pixelCount = width * height;
        if (sourcePixels == null || sourcePixels.length < pixelCount) {
            sourcePixels = new int[pixelCount];
        }
        if (rotatedPixels == null || rotatedPixels.length < pixelCount) {
            rotatedPixels = new int[pixelCount];
        }

        Rgba8888Converter.toArgb(
                planeBytes,
                width,
                height,
                plane.getRowStride(),
                plane.getPixelStride(),
                sourcePixels);
        Rect cropRect = image.getCropRect();
        int cropWidth = cropRect.width();
        int cropHeight = cropRect.height();
        int cropPixelCount = cropWidth * cropHeight;
        if (croppedPixels == null || croppedPixels.length < cropPixelCount) {
            croppedPixels = new int[cropPixelCount];
        }
        FrameCropper.cropInto(
                sourcePixels,
                width,
                height,
                cropRect.left,
                cropRect.top,
                cropRect.right,
                cropRect.bottom,
                croppedPixels);
        int rotationDegrees = image.getImageInfo().getRotationDegrees();
        FrameRotation.rotateInto(
                croppedPixels,
                cropWidth,
                cropHeight,
                rotationDegrees,
                rotatedPixels);

        int rotatedWidth = rotationDegrees == 90 || rotationDegrees == 270
                ? cropHeight : cropWidth;
        int rotatedHeight = rotationDegrees == 90 || rotationDegrees == 270
                ? cropWidth : cropHeight;
        if (frameBitmap == null
                || frameBitmap.getWidth() != rotatedWidth
                || frameBitmap.getHeight() != rotatedHeight) {
            frameBitmap = findOrCreateFrameBitmap(rotatedWidth, rotatedHeight);
        }
        frameBitmap.setPixels(
                rotatedPixels,
                0,
                rotatedWidth,
                0,
                0,
                rotatedWidth,
                rotatedHeight);
        return frameBitmap;
    }

    private Bitmap findOrCreateFrameBitmap(int width, int height) {
        for (Bitmap bitmap : frameBitmaps) {
            if (!bitmap.isRecycled()
                    && bitmap.getWidth() == width
                    && bitmap.getHeight() == height) {
                return bitmap;
            }
        }
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        frameBitmaps.add(bitmap);
        return bitmap;
    }

    private void closeInferenceResources() {
        FastenerDetector currentDetector = detector;
        detector = null;
        if (currentDetector != null) {
            currentDetector.close();
        }
        OnnxWitnessStateEstimator currentStateEstimator = stateEstimator;
        stateEstimator = null;
        if (currentStateEstimator != null) {
            currentStateEstimator.close();
        }
        for (Bitmap bitmap : frameBitmaps) {
            if (!bitmap.isRecycled()) {
                bitmap.recycle();
            }
        }
        frameBitmaps.clear();
        frameBitmap = null;
        planeBytes = null;
        sourcePixels = null;
        croppedPixels = null;
        rotatedPixels = null;
    }

    private void postResult(
            OnnxFastenerDetector.DetectionResult result,
            double approximateFps,
            Bitmap frozenDetectionFrame) {
        Log.i(TAG, String.format(
                java.util.Locale.US,
                "detections=%d total=%.1fms preprocess=%.1fms inference=%.1fms postprocess=%.1fms fps=%.2f",
                result.getDetections().size(),
                result.getLatencyMillis(),
                result.getPreprocessMillis(),
                result.getInferenceMillis(),
                result.getPostprocessMillis(),
                approximateFps));
        String metrics = getString(
                R.string.live_metrics_format,
                result.getDetections().size(),
                result.getLatencyMillis(),
                approximateFps);
        runOnUiThread(() -> {
            if (destroyed || reviewInProgress) {
                frozenDetectionFrame.recycle();
                return;
            }
            modelStatusView.setText(R.string.live_model_ready);
            metricsView.setText(metrics);
            overlayView.setDetections(
                    result.getDetections(),
                    result.getOriginalWidth(),
                    result.getOriginalHeight());
            Bitmap previousDetectionFrame = detectionFrame;
            detectionFrame = frozenDetectionFrame;
            if (previousDetectionFrame != null && !previousDetectionFrame.isRecycled()) {
                previousDetectionFrame.recycle();
            }
        });
    }

    private void onDetectionTapped(Detection detection) {
        if (destroyed || reviewInProgress || detectionFrame == null) {
            return;
        }
        reviewInProgress = true;
        Bitmap selectedFrame = detectionFrame.copy(Bitmap.Config.ARGB_8888, false);
        if (selectedFrame == null) {
            showStateReviewDialog(null, null);
            return;
        }
        inferenceExecutor.execute(() -> estimateTappedDetection(selectedFrame, detection));
    }

    private void estimateTappedDetection(Bitmap selectedFrame, Detection detection) {
        Bitmap roiPreview = null;
        WitnessStateEstimate estimate = null;
        try {
            SquareRoi roi = SquareRoi.fromDetection(
                    detection, selectedFrame.getWidth(), selectedFrame.getHeight());
            roiPreview = Bitmap.createBitmap(
                    selectedFrame,
                    roi.getLeft(),
                    roi.getTop(),
                    roi.getSide(),
                    roi.getSide());
            OnnxWitnessStateEstimator currentEstimator = stateEstimator;
            if (currentEstimator == null || !currentEstimator.isReady()) {
                throw new IllegalStateException("experimental witness state estimator unavailable");
            }
            estimate = currentEstimator.estimate(selectedFrame, roi);
        } catch (Exception | LinkageError exception) {
            Log.w(TAG, "Experimental witness state estimate failed closed", exception);
        } finally {
            selectedFrame.recycle();
        }
        Bitmap finalRoiPreview = roiPreview;
        WitnessStateEstimate finalEstimate = estimate;
        runOnUiThread(() -> showStateReviewDialog(finalRoiPreview, finalEstimate));
    }

    private void showStateReviewDialog(
            Bitmap roiPreview, WitnessStateEstimate estimate) {
        if (destroyed) {
            if (roiPreview != null && !roiPreview.isRecycled()) {
                roiPreview.recycle();
            }
            reviewInProgress = false;
            return;
        }
        LinearLayout content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL);
        int padding = dp(20);
        content.setPadding(padding, dp(8), padding, 0);

        if (roiPreview != null) {
            ImageView image = new ImageView(this);
            image.setAdjustViewBounds(true);
            image.setScaleType(ImageView.ScaleType.FIT_CENTER);
            image.setMinimumHeight(dp(180));
            image.setMaxHeight(dp(300));
            image.setImageBitmap(roiPreview);
            content.addView(image, new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT));
        }

        TextView details = new TextView(this);
        details.setTextSize(16f);
        details.setTextColor(0xFF202124);
        details.setGravity(Gravity.START);
        LinearLayout.LayoutParams detailParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        detailParams.topMargin = dp(14);
        if (estimate == null) {
            details.setText(R.string.live_state_unavailable);
        } else {
            details.setText(getString(
                    R.string.live_state_result_format,
                    estimate.getAngle().getPointEstimateDegrees(),
                    estimate.getAngle().getLowerDegrees(),
                    estimate.getAngle().getUpperDegrees(),
                    thresholdBucket(estimate),
                    estimate.getInferenceMillis(),
                    reviewAdvice(estimate)));
        }
        content.addView(details, detailParams);

        TextView warning = new TextView(this);
        warning.setText(R.string.live_state_experimental_warning);
        warning.setTextSize(17f);
        warning.setTextColor(0xFFC62828);
        warning.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams warningParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        warningParams.topMargin = dp(18);
        content.addView(warning, warningParams);

        AlertDialog dialog = new AlertDialog.Builder(this)
                .setTitle(R.string.live_state_experimental_title)
                .setView(content)
                .setPositiveButton(R.string.live_state_confirm, null)
                .create();
        activeReviewDialog = dialog;
        dialog.setCanceledOnTouchOutside(false);
        dialog.setOnDismissListener(ignored -> {
            activeReviewDialog = null;
            reviewInProgress = false;
            if (roiPreview != null && !roiPreview.isRecycled()) {
                roiPreview.recycle();
            }
        });
        dialog.show();
    }

    private String thresholdBucket(WitnessStateEstimate estimate) {
        if (estimate.getReviewHint() == WitnessReviewHint.LIKELY_ALIGNED) {
            return getString(R.string.live_state_bucket_likely_aligned);
        }
        if ("POINT_ANGLE_SECOND_VIEW_REQUIRED".equals(estimate.getReviewReason())) {
            return getString(R.string.live_state_bucket_high);
        }
        return getString(R.string.live_state_bucket_review);
    }

    private String reviewAdvice(WitnessStateEstimate estimate) {
        if (estimate.getReviewHint() == WitnessReviewHint.LIKELY_ALIGNED) {
            return getString(R.string.live_state_review_likely_aligned);
        }
        if ("POINT_ANGLE_SECOND_VIEW_REQUIRED".equals(estimate.getReviewReason())) {
            return getString(R.string.live_state_review_second_view);
        }
        return getString(R.string.live_state_review_retake);
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
