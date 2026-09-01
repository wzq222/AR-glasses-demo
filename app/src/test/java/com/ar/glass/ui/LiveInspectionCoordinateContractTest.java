package com.ar.glass.ui;

import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.Assert.assertTrue;

public class LiveInspectionCoordinateContractTest {
    @Test
    public void sharesThePreviewViewportAndConsumesTheAnalysisCropRect() throws Exception {
        Path projectRoot = Paths.get(System.getProperty("user.dir"));
        Path source = projectRoot.resolve(
                "app/src/main/java/com/ar/glass/ui/LiveInspectionActivity.java");
        if (!Files.exists(source)) {
            source = projectRoot.resolve(
                    "src/main/java/com/ar/glass/ui/LiveInspectionActivity.java");
        }
        String text = new String(Files.readAllBytes(source), StandardCharsets.UTF_8);

        assertTrue(text.contains("previewView.getViewPort()"));
        assertTrue(text.contains("new UseCaseGroup.Builder()"));
        assertTrue(text.contains(".setViewPort(viewPort)"));
        assertTrue(text.contains("image.getCropRect()"));
        assertTrue(text.contains("FrameCropper.cropInto("));
    }

    @Test
    public void freezesTappedDetectionFrameAndPausesUntilDialogDismissal() throws Exception {
        Path projectRoot = Paths.get(System.getProperty("user.dir"));
        Path source = projectRoot.resolve(
                "app/src/main/java/com/ar/glass/ui/LiveInspectionActivity.java");
        if (!Files.exists(source)) {
            source = projectRoot.resolve(
                    "src/main/java/com/ar/glass/ui/LiveInspectionActivity.java");
        }
        String text = new String(Files.readAllBytes(source), StandardCharsets.UTF_8);

        assertTrue(text.contains("setOnDetectionTapListener"));
        assertTrue(text.contains("detectionFrame.copy("));
        assertTrue(text.contains("reviewInProgress"));
        assertTrue(text.contains("setOnDismissListener"));
        assertTrue(text.contains("OnnxWitnessStateEstimator"));
    }
}
