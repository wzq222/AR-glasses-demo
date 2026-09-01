package com.ar.glass.vision.realtime;

import android.graphics.Bitmap;

import java.io.Closeable;

public interface FastenerDetector extends Closeable {
    boolean isReady();

    String getInitializationError();

    OnnxFastenerDetector.DetectionResult detect(Bitmap bitmap) throws Exception;

    @Override
    void close();
}
