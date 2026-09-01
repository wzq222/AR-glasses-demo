package com.ar.glass.vision.realtime;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

public class DetectionHitTesterTest {
    @Test
    public void fillCenterSelectsSmallestContainingCandidate() {
        Detection large = new Detection(100f, 100f, 400f, 400f, 0.8f, 0);
        Detection small = new Detection(180f, 180f, 260f, 260f, 0.7f, 0);
        List<Detection> detections = Arrays.asList(large, small);

        Detection selected = DetectionHitTester.smallestContainingFillCenter(
                detections, 640, 480, 1000, 1000, 343.75f, 458.33f);

        assertEquals(small, selected);
    }

    @Test
    public void fillCenterAccountsForCroppedFrameEdges() {
        Detection leftEdge = new Detection(0f, 100f, 120f, 200f, 0.8f, 0);

        Detection selected = DetectionHitTester.smallestContainingFillCenter(
                Arrays.asList(leftEdge), 640, 480, 1000, 1000, -50f, 300f);

        assertEquals(leftEdge, selected);
    }

    @Test
    public void returnsNullOutsideEveryCandidate() {
        Detection candidate = new Detection(100f, 100f, 200f, 200f, 0.8f, 0);

        assertNull(DetectionHitTester.smallestContainingFillCenter(
                Arrays.asList(candidate), 640, 480, 1000, 1000, 900f, 900f));
    }
}
