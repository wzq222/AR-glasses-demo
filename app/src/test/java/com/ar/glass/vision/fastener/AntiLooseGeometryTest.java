package com.ar.glass.vision.fastener;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class AntiLooseGeometryTest {
    private static VisionPoint p(float x, float y, float confidence) {
        return new VisionPoint(x, y, confidence);
    }

    @Test
    public void alignedSegmentsAreNormal() {
        GeometryDecision result = AntiLooseGeometry.evaluate(
                p(0, 0, 0.95f), p(10, 0, 0.95f),
                p(12, 0, 0.95f), p(22, 0, 0.95f),
                100f, new GeometryThresholds(10f, 0.05f, 0.05f));

        assertEquals(FastenerState.NORMAL, result.getState());
        assertEquals("GEOMETRY_WITHIN_THRESHOLDS", result.getReason());
    }

    @Test
    public void reversedSegmentDirectionIsStillAligned() {
        GeometryDecision result = AntiLooseGeometry.evaluate(
                p(10, 0, 0.95f), p(0, 0, 0.95f),
                p(22, 0, 0.95f), p(12, 0, 0.95f),
                100f, new GeometryThresholds(10f, 0.05f, 0.05f));

        assertEquals(FastenerState.NORMAL, result.getState());
    }

    @Test
    public void excessiveAngleIsLoose() {
        GeometryDecision result = AntiLooseGeometry.evaluate(
                p(0, 0, 0.95f), p(10, 0, 0.95f),
                p(12, 0, 0.95f), p(12, 10, 0.95f),
                100f, new GeometryThresholds(10f, 0.20f, 0.20f));

        assertEquals(FastenerState.LOOSE, result.getState());
        assertEquals("ANGLE_EXCEEDED", result.getReason());
    }

    @Test
    public void lowConfidenceIsUncertain() {
        GeometryDecision result = AntiLooseGeometry.evaluate(
                p(0, 0, 0.4f), p(10, 0, 0.95f),
                p(12, 0, 0.95f), p(22, 0, 0.95f),
                100f, new GeometryThresholds(10f, 0.05f, 0.05f));

        assertEquals(FastenerState.UNCERTAIN, result.getState());
        assertEquals("KEYPOINT_CONFIDENCE_LOW", result.getReason());
    }

    @Test
    public void productionWithoutCalibratedThresholdsIsUncertain() {
        GeometryDecision result = AntiLooseGeometry.evaluate(
                p(0, 0, 0.95f), p(10, 0, 0.95f),
                p(12, 0, 0.95f), p(22, 0, 0.95f),
                100f, GeometryThresholds.uncalibrated());

        assertEquals(FastenerState.UNCERTAIN, result.getState());
        assertEquals("THRESHOLDS_UNCALIBRATED", result.getReason());
    }
}
