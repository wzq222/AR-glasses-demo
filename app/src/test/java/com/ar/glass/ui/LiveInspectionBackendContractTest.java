package com.ar.glass.ui;

import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.Assert.assertTrue;

public class LiveInspectionBackendContractTest {
    @Test
    public void liveInspectionUsesTheBuildSelectedSharedDetector() throws Exception {
        Path projectRoot = Paths.get(System.getProperty("user.dir"));
        Path source = projectRoot.resolve("app/src/main/java/com/ar/glass/ui/LiveInspectionActivity.java");
        if (!Files.exists(source)) {
            source = projectRoot.resolve("src/main/java/com/ar/glass/ui/LiveInspectionActivity.java");
        }
        String text = new String(Files.readAllBytes(source), StandardCharsets.UTF_8);

        assertTrue(text.contains("private volatile FastenerDetector detector"));
        assertTrue(text.contains("DetectorFactory.create"));
        assertTrue(text.contains("BuildConfig.DETECTOR_BACKEND"));
        assertTrue(text.contains("BuildConfig.NCNN_VULKAN"));
        assertTrue(text.contains("BuildConfig.NCNN_VULKAN_FP16"));
        assertTrue(text.contains("private volatile boolean inferenceFailed"));
        assertTrue(text.contains("overlayView.clearDetections()"));
        assertTrue(text.contains("inferenceFailed = true"));
    }
}
