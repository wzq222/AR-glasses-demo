package com.ar.glass.ui;

import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.Assert.assertTrue;

public class DetectionOverlayContractTest {
    @Test
    public void delegatesFillCenterTapSelectionToPureHitTester() throws Exception {
        Path projectRoot = Paths.get(System.getProperty("user.dir"));
        Path source = projectRoot.resolve(
                "app/src/main/java/com/ar/glass/ui/DetectionOverlayView.java");
        if (!Files.exists(source)) {
            source = projectRoot.resolve(
                    "src/main/java/com/ar/glass/ui/DetectionOverlayView.java");
        }
        String text = new String(Files.readAllBytes(source), StandardCharsets.UTF_8);

        assertTrue(text.contains("interface OnDetectionTapListener"));
        assertTrue(text.contains("setOnDetectionTapListener"));
        assertTrue(text.contains("DetectionHitTester.smallestContainingFillCenter"));
    }
}
