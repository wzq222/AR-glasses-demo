package com.ar.glass.ui;

import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.Assert.assertTrue;

public class DetectionRenderingContractTest {
    @Test
    public void previewShowsBoltIdentityStateAndFourPointGeometry() throws Exception {
        String overlay = readProjectFile(
                "app/src/main/java/com/ar/glass/vision/ui/BoxOverlay.java");
        String gallery = readProjectFile(
                "app/src/main/java/com/ar/glass/ui/ImageViewerActivity.java");

        assertTrue(overlay.contains("InspectionPresentation.stateLabel"));
        assertTrue(overlay.contains("drawWitnessSegments"));
        assertTrue(gallery.contains("InspectionPresentation.stateLabel"));
        assertTrue(gallery.contains("drawWitnessGeometry"));
        assertTrue(gallery.contains("canvas.drawLine"));
        assertTrue(gallery.contains("canvas.drawCircle"));
    }

    private static String readProjectFile(String relativePath) throws Exception {
        Path projectRoot = Paths.get(System.getProperty("user.dir"));
        Path file = projectRoot.resolve(relativePath);
        if (!Files.exists(file) && relativePath.startsWith("app/")) {
            file = projectRoot.resolve(relativePath.substring("app/".length()));
        }
        return new String(Files.readAllBytes(file), StandardCharsets.UTF_8);
    }
}
