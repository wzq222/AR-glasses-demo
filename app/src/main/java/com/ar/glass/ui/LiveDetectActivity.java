package com.ar.glass.ui;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import com.ar.glass.R;
import com.ar.glass.vision.MarkedPointDetectorHolder;
import com.ar.glass.vision.ui.BoxOverlay;

import com.google.common.util.concurrent.ListenableFuture;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 手机摄像头实时检测：CameraX ImageAnalysis 逐帧送检（节流），预览 + 检测框叠加 + FPS。
 * 检测模型跟随 DetectionRouter（紧固件防松标记 / 通用螺丝检测）。
 */
public class LiveDetectActivity extends AppCompatActivity {

    private static final String TAG = "LiveDetect";
    private static final int REQUEST_CAMERA = 301;
    /** 检测节流：两帧检测之间的最小间隔 ms */
    private static final long DETECT_INTERVAL_MS = 120;

    private PreviewView previewView;
    private BoxOverlay overlay;
    private TextView tvStatus;
    private View btnBack;

    private ExecutorService analysisExecutor;
    private final AtomicBoolean detecting = new AtomicBoolean(false);
    private volatile long lastDetectMs;
    private int frameW, frameH;
    private long fpsWindowStart;
    private int fpsFrames;
    private double fps;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_live_detect);
        previewView = findViewById(R.id.previewView);
        overlay = findViewById(R.id.liveOverlay);
        tvStatus = findViewById(R.id.tvLiveStatus);
        btnBack = findViewById(R.id.btnLiveBack);
        btnBack.setOnClickListener(v -> finish());
        analysisExecutor = Executors.newSingleThreadExecutor();

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA);
        } else {
            startCamera();
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> future =
                ProcessCameraProvider.getInstance(this);
        future.addListener(() -> {
            try {
                ProcessCameraProvider provider = future.get();
                Preview preview = new Preview.Builder()
                        .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                        .build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();
                analysis.setAnalyzer(analysisExecutor, this::analyzeFrame);

                provider.unbindAll();
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA,
                        preview, analysis);
                tvStatus.setText("实时检测中…");
            } catch (Exception e) {
                Log.e(TAG, "startCamera error", e);
                Toast.makeText(this, "相机启动失败：" + e.getMessage(), Toast.LENGTH_LONG).show();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void analyzeFrame(ImageProxy image) {
        try {
            long now = System.currentTimeMillis();
            if (detecting.get() || now - lastDetectMs < DETECT_INTERVAL_MS) {
                return;
            }
            detecting.set(true);
            lastDetectMs = now;

            Bitmap bitmap = image.toBitmap();
            int rotation = image.getImageInfo().getRotationDegrees();
            if (rotation != 0) {
                android.graphics.Matrix m = new android.graphics.Matrix();
                m.postRotate(rotation);
                bitmap = Bitmap.createBitmap(bitmap, 0, 0,
                        bitmap.getWidth(), bitmap.getHeight(), m, true);
            }
            frameW = bitmap.getWidth();
            frameH = bitmap.getHeight();
            Bitmap frame = bitmap;

            List<com.ar.glass.vision.YoloDetector.Detection> dets;
            long t0 = System.currentTimeMillis();
            MarkedPointDetectorHolder.Result r =
                    MarkedPointDetectorHolder.detect(this, frame);
            dets = r != null ? r.detections : null;
            long ms = System.currentTimeMillis() - t0;

            // FPS 统计
            fpsFrames++;
            if (fpsWindowStart == 0) fpsWindowStart = now;
            if (now - fpsWindowStart >= 1000) {
                fps = fpsFrames * 1000.0 / (now - fpsWindowStart);
                fpsFrames = 0;
                fpsWindowStart = now;
            }
            final int count = dets != null ? dets.size() : -1;
            final long inferMs = ms;
            runOnUiThread(() -> {
                if (dets == null) {
                    tvStatus.setText("防松标记模型未就绪");
                } else {
                    overlay.setResults(dets, frameW, frameH);
                    tvStatus.setText("[防松标记] "
                            + count + " 个检查点 · " + inferMs + "ms · "
                            + String.format(java.util.Locale.US, "%.1f", fps) + " FPS");
                }
            });
        } catch (Throwable e) {
            Log.e(TAG, "analyzeFrame error", e);
        } finally {
            detecting.set(false);
            image.close();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CAMERA) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera();
            } else {
                Toast.makeText(this, "需要相机权限才能实时检测", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (analysisExecutor != null) analysisExecutor.shutdown();
    }
}
