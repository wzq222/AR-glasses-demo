package com.ar.glass.vision.realtime;

import java.util.Locale;

public enum DetectorBackend {
    ONNX,
    NCNN;

    public static DetectorBackend fromBuildValue(String value) {
        if (value == null || value.trim().isEmpty()) {
            return ONNX;
        }
        String normalized = value.trim().toUpperCase(Locale.US);
        for (DetectorBackend backend : values()) {
            if (backend.name().equals(normalized)) {
                return backend;
            }
        }
        throw new IllegalArgumentException("Unsupported detector backend: " + value);
    }
}
