package com.ar.glass.ui;

import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.Assert.assertTrue;

public class MobileDetectorBenchmarkContractTest {
    @Test
    public void debugHarnessRunsTheSelectedBackendOnAnInternalImage() throws Exception {
        Path projectRoot = Paths.get(System.getProperty("user.dir"));
        Path source = projectRoot.resolve(
                "app/src/debug/java/com/ar/glass/ui/MobileDetectorBenchmarkActivity.java");
        if (!Files.exists(source)) {
            source = projectRoot.resolve(
                    "src/debug/java/com/ar/glass/ui/MobileDetectorBenchmarkActivity.java");
        }
        String text = new String(Files.readAllBytes(source), StandardCharsets.UTF_8);

        assertTrue(text.contains("DetectorFactory.create"));
        assertTrue(text.contains("BuildConfig.DETECTOR_BACKEND"));
        assertTrue(text.contains("BuildConfig.MARKED_POINT_VERIFIER_ENABLED"));
        assertTrue(text.contains("getFileStreamPath"));
        assertTrue(text.contains("getStringExtra(\"directory\")"));
        assertTrue(text.contains("getBooleanExtra(\"detailed_boxes\", false)"));
        assertTrue(text.contains("getStringExtra(\"run_token\")"));
        assertTrue(text.contains("box image=%s index=%d class=%d"));
        assertTrue(text.contains("run_token=%s"));
        assertTrue(text.contains("complete images=%d run_token=%s"));
        assertTrue(text.contains("DetectorBenchmark"));
    }
}
