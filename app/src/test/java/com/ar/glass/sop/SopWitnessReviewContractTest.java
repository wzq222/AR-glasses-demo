package com.ar.glass.sop;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.junit.Test;

public class SopWitnessReviewContractTest {
    @Test
    public void fastenerStepPersistsAiTriageAndPointLevelHumanReview() throws Exception {
        String source = readProjectFile(
                "app/src/main/java/com/ar/glass/sop/SopActivity.java");

        assertTrue(source.contains("MarkedPointDetectorHolder.detect(this, bitmap)"));
        assertTrue(source.contains("value.put(\"ai_triage\""));
        assertTrue(source.contains("value.put(\"point_results\""));
        assertTrue(source.contains("point.put(\"requires_human_confirmation\", true)"));
        assertTrue(source.contains("pointResultMaps.get(index).put(\"human_decision\""));
        assertTrue(source.contains("new String[]{\"请选择逐点结论\", \"确认正常\", \"疑似松动\", \"无法判断/重拍\"}"));
        assertTrue(source.contains("value.put(\"human_review_complete\", true)"));
        assertFalse(source.contains(
                "value.put(\"marked_point_count\", detection.detections.size());\n"
                        + "                        value.put(\"state\", \"INSUFFICIENT\");"));
    }

    @Test
    public void evidencePreviewIncludesTheWitnessOverlayAndReviewList() throws Exception {
        String layout = readProjectFile("app/src/main/res/layout/activity_sop.xml");

        assertTrue(layout.contains("android:id=\"@+id/overlaySopEvidence\""));
        assertTrue(layout.contains("com.ar.glass.vision.ui.BoxOverlay"));
        assertTrue(layout.contains("android:id=\"@+id/markedPointReviewList\""));
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
