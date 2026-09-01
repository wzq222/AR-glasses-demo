package com.ar.glass.ui;

import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertFalse;

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

    @Test
    public void enablesCandidateTapOnlyAfterExperimentalEstimatorIsReady() throws Exception {
        String text = readLiveInspectionSource();
        String onCreate = methodBody(text, "protected void onCreate(Bundle savedInstanceState)");
        String initialize = methodBody(text, "private void initializeDetectorInBackground()");

        assertFalse(onCreate.contains("setOnDetectionTapListener"));
        assertTrue(initialize.contains("WitnessStateInteractionPolicy.forRuntime("));
        assertTrue(initialize.contains("policy.canTapCandidate()"));
        assertTrue(initialize.contains("overlayView.setOnDetectionTapListener("));
        assertTrue(initialize.contains("Experimental witness state estimator ready"));
    }

    @Test
    public void previewsAndEstimatesThePotentiallyRectangularTrainingCrop() throws Exception {
        String text = readLiveInspectionSource();

        assertTrue(text.contains("WitnessRoi.fromDetection("));
        assertTrue(text.contains("roi.getWidth()"));
        assertTrue(text.contains("roi.getHeight()"));
    }

    private static String readLiveInspectionSource() throws Exception {
        Path projectRoot = Paths.get(System.getProperty("user.dir"));
        Path source = projectRoot.resolve(
                "app/src/main/java/com/ar/glass/ui/LiveInspectionActivity.java");
        if (!Files.exists(source)) {
            source = projectRoot.resolve(
                    "src/main/java/com/ar/glass/ui/LiveInspectionActivity.java");
        }
        return new String(Files.readAllBytes(source), StandardCharsets.UTF_8);
    }

    private static String methodBody(String source, String signature) {
        int signatureStart = source.indexOf(signature);
        assertTrue("Missing method: " + signature, signatureStart >= 0);
        int bodyStart = source.indexOf('{', signatureStart);
        int depth = 0;
        for (int index = bodyStart; index < source.length(); index++) {
            char character = source.charAt(index);
            if (character == '{') {
                depth++;
            } else if (character == '}') {
                depth--;
                if (depth == 0) {
                    return source.substring(bodyStart, index + 1);
                }
            }
        }
        throw new AssertionError("Unterminated method: " + signature);
    }
}
