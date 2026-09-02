package com.ar.glass.ui;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;

import com.ar.glass.BuildConfig;
import com.ar.glass.vision.realtime.Detection;
import com.ar.glass.vision.realtime.DetectorFactory;
import com.ar.glass.vision.realtime.FastenerDetector;
import com.ar.glass.vision.realtime.OnnxFastenerDetector;

import java.io.File;
import java.util.List;
import java.util.Locale;
import java.util.Arrays;
import java.util.regex.Pattern;

public final class MobileDetectorBenchmarkActivity extends Activity {
    private static final String TAG = "DetectorBenchmark";
    private static final String DEFAULT_IMAGE_NAME = "benchmark.jpg";
    private static final int RUN_COUNT = 6;
    private static final Pattern RUN_TOKEN_PATTERN =
            Pattern.compile("[A-Za-z0-9_-]{1,64}");

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        new Thread(this::runBenchmark, "mobile-detector-benchmark").start();
    }

    private void runBenchmark() {
        boolean detailedBoxes = getIntent().getBooleanExtra("detailed_boxes", false);
        String runToken = resolveRunToken(getIntent().getStringExtra("run_token"));
        String directoryName = getIntent().getStringExtra("directory");
        if (directoryName != null && !directoryName.trim().isEmpty()) {
            runDirectoryBenchmark(directoryName.trim(), detailedBoxes, runToken);
            return;
        }
        String imageName = getIntent().getStringExtra("image");
        if (imageName == null || imageName.trim().isEmpty()) {
            imageName = DEFAULT_IMAGE_NAME;
        }
        FastenerDetector detector = DetectorFactory.create(
                getApplicationContext(),
                BuildConfig.DETECTOR_BACKEND,
                BuildConfig.NCNN_CONFIDENCE_THRESHOLD,
                BuildConfig.NCNN_VULKAN,
                BuildConfig.NCNN_VULKAN_FP16,
                BuildConfig.MARKED_POINT_VERIFIER_ENABLED,
                BuildConfig.MARKED_POINT_VERIFIER_NNAPI,
                BuildConfig.MARKED_POINT_VERIFIER_XNNPACK);
        try {
            if (!detector.isReady()) {
                Log.e(TAG, "backend=" + BuildConfig.DETECTOR_BACKEND
                        + " initialization=" + detector.getInitializationError());
                return;
            }
            boolean completed = benchmarkImage(
                    detector, getFileStreamPath(imageName), RUN_COUNT, detailedBoxes, runToken);
            Log.i(TAG, String.format(
                    Locale.US,
                    "complete images=%d run_token=%s",
                    completed ? 1 : 0,
                    runToken));
        } catch (Exception | LinkageError exception) {
            Log.e(TAG, "benchmark failed", exception);
        } finally {
            detector.close();
            finish();
        }
    }

    private void runDirectoryBenchmark(
            String directoryName, boolean detailedBoxes, String runToken) {
        FastenerDetector detector = DetectorFactory.create(
                getApplicationContext(),
                BuildConfig.DETECTOR_BACKEND,
                BuildConfig.NCNN_CONFIDENCE_THRESHOLD,
                BuildConfig.NCNN_VULKAN,
                BuildConfig.NCNN_VULKAN_FP16,
                BuildConfig.MARKED_POINT_VERIFIER_ENABLED,
                BuildConfig.MARKED_POINT_VERIFIER_NNAPI,
                BuildConfig.MARKED_POINT_VERIFIER_XNNPACK);
        try {
            File directory = new File(getFilesDir(), directoryName);
            File[] images = directory.listFiles(file -> {
                String name = file.getName().toLowerCase(Locale.US);
                return file.isFile() && (name.endsWith(".jpg") || name.endsWith(".png"));
            });
            if (images == null || images.length == 0) {
                Log.e(TAG, "no benchmark images in internal directory=" + directoryName);
                return;
            }
            if (!detector.isReady()) {
                Log.e(TAG, "backend=" + BuildConfig.DETECTOR_BACKEND
                        + " initialization=" + detector.getInitializationError());
                return;
            }
            Arrays.sort(images, (first, second) -> first.getName().compareTo(second.getName()));
            int completedImages = 0;
            for (File image : images) {
                if (benchmarkImage(detector, image, 1, detailedBoxes, runToken)) {
                    completedImages++;
                }
            }
            Log.i(TAG, String.format(
                    Locale.US,
                    "complete images=%d run_token=%s",
                    completedImages,
                    runToken));
        } catch (Exception | LinkageError exception) {
            Log.e(TAG, "directory benchmark failed", exception);
        } finally {
            detector.close();
            finish();
        }
    }

    private static boolean benchmarkImage(
            FastenerDetector detector,
            File imageFile,
            int runCount,
            boolean detailedBoxes,
            String runToken) throws Exception {
        Bitmap bitmap = BitmapFactory.decodeFile(imageFile.getAbsolutePath());
        if (bitmap == null) {
            Log.e(TAG, "unable to decode internal image=" + imageFile.getName()
                    + " run_token=" + runToken);
            return false;
        }
        try {
            Log.i(TAG, "start backend=" + BuildConfig.DETECTOR_BACKEND
                    + " image=" + imageFile.getName()
                    + " width=" + bitmap.getWidth()
                    + " height=" + bitmap.getHeight()
                    + " run_token=" + runToken);
            for (int run = 0; run < runCount; run++) {
                OnnxFastenerDetector.DetectionResult result = detector.detect(bitmap);
                Log.i(TAG, String.format(
                        Locale.US,
                        "image=%s run=%d detections=%d total=%.3fms preprocess=%.3fms "
                                + "inference=%.3fms postprocess=%.3fms run_token=%s",
                        imageFile.getName(),
                        run,
                        result.getDetections().size(),
                        result.getLatencyMillis(),
                        result.getPreprocessMillis(),
                        result.getInferenceMillis(),
                        result.getPostprocessMillis(),
                        runToken));
                if (detailedBoxes) {
                    logDetections(imageFile.getName(), result.getDetections(), runToken);
                }
            }
            return true;
        } finally {
            bitmap.recycle();
        }
    }

    private static void logDetections(
            String imageName, List<Detection> detections, String runToken) {
        for (int index = 0; index < detections.size(); index++) {
            Detection detection = detections.get(index);
            Log.i(TAG, String.format(
                    Locale.US,
                    "box image=%s index=%d class=%d score=%.9f "
                            + "left=%.6f top=%.6f right=%.6f bottom=%.6f run_token=%s",
                    imageName,
                    index,
                    detection.getClassId(),
                    detection.getConfidence(),
                    detection.getLeft(),
                    detection.getTop(),
                    detection.getRight(),
                    detection.getBottom(),
                    runToken));
        }
    }

    private static String resolveRunToken(String requestedToken) {
        if (requestedToken != null && RUN_TOKEN_PATTERN.matcher(requestedToken).matches()) {
            return requestedToken;
        }
        return "manual-" + Long.toHexString(android.os.SystemClock.elapsedRealtime());
    }

}
