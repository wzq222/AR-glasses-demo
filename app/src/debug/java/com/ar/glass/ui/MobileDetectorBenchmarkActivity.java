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

public final class MobileDetectorBenchmarkActivity extends Activity {
    private static final String TAG = "DetectorBenchmark";
    private static final String DEFAULT_IMAGE_NAME = "benchmark.jpg";
    private static final int RUN_COUNT = 6;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        new Thread(this::runBenchmark, "mobile-detector-benchmark").start();
    }

    private void runBenchmark() {
        String directoryName = getIntent().getStringExtra("directory");
        if (directoryName != null && !directoryName.trim().isEmpty()) {
            runDirectoryBenchmark(directoryName.trim());
            return;
        }
        String imageName = getIntent().getStringExtra("image");
        if (imageName == null || imageName.trim().isEmpty()) {
            imageName = DEFAULT_IMAGE_NAME;
        }
        FastenerDetector detector = DetectorFactory.create(
                getApplicationContext(),
                BuildConfig.DETECTOR_BACKEND,
                BuildConfig.NCNN_CONFIDENCE_THRESHOLD);
        try {
            if (!detector.isReady()) {
                Log.e(TAG, "backend=" + BuildConfig.DETECTOR_BACKEND
                        + " initialization=" + detector.getInitializationError());
                return;
            }
            benchmarkImage(detector, getFileStreamPath(imageName), RUN_COUNT);
        } catch (Exception | LinkageError exception) {
            Log.e(TAG, "benchmark failed", exception);
        } finally {
            detector.close();
            finish();
        }
    }

    private void runDirectoryBenchmark(String directoryName) {
        FastenerDetector detector = DetectorFactory.create(
                getApplicationContext(),
                BuildConfig.DETECTOR_BACKEND,
                BuildConfig.NCNN_CONFIDENCE_THRESHOLD);
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
            for (File image : images) {
                benchmarkImage(detector, image, 1);
            }
        } catch (Exception | LinkageError exception) {
            Log.e(TAG, "directory benchmark failed", exception);
        } finally {
            detector.close();
            finish();
        }
    }

    private static void benchmarkImage(
            FastenerDetector detector, File imageFile, int runCount) throws Exception {
        Bitmap bitmap = BitmapFactory.decodeFile(imageFile.getAbsolutePath());
        if (bitmap == null) {
            Log.e(TAG, "unable to decode internal image=" + imageFile.getName());
            return;
        }
        try {
            Log.i(TAG, "start backend=" + BuildConfig.DETECTOR_BACKEND
                    + " image=" + imageFile.getName()
                    + " width=" + bitmap.getWidth()
                    + " height=" + bitmap.getHeight());
            for (int run = 0; run < runCount; run++) {
                OnnxFastenerDetector.DetectionResult result = detector.detect(bitmap);
                Log.i(TAG, String.format(
                        Locale.US,
                        "image=%s run=%d detections=%d total=%.3fms preprocess=%.3fms "
                                + "inference=%.3fms postprocess=%.3fms boxes=%s",
                        imageFile.getName(),
                        run,
                        result.getDetections().size(),
                        result.getLatencyMillis(),
                        result.getPreprocessMillis(),
                        result.getInferenceMillis(),
                        result.getPostprocessMillis(),
                        formatDetections(result.getDetections())));
            }
        } finally {
            bitmap.recycle();
        }
    }

    private static String formatDetections(List<Detection> detections) {
        StringBuilder output = new StringBuilder();
        output.append('[');
        for (int index = 0; index < detections.size(); index++) {
            if (index > 0) {
                output.append(';');
            }
            Detection detection = detections.get(index);
            output.append(String.format(
                    Locale.US,
                    "%d,%.6f,%.3f,%.3f,%.3f,%.3f",
                    detection.getClassId(),
                    detection.getConfidence(),
                    detection.getLeft(),
                    detection.getTop(),
                    detection.getRight(),
                    detection.getBottom()));
        }
        output.append(']');
        return output.toString();
    }
}
