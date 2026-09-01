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
        assertTrue(text.contains("getFileStreamPath"));
        assertTrue(text.contains("getStringExtra(\"directory\")"));
        assertTrue(text.contains("DetectorBenchmark"));
    }
}
