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
                FastenerTopology.NUT_PLATE, WitnessMarkRole.BRIDGES_MOVING_FIXED, true,
                p(0, 0, 0.95f), p(10, 0, 0.95f),
                p(12, 0, 0.95f), p(22, 0, 0.95f),
                100f, new GeometryThresholds(10f, 0.05f, 0.05f));

        assertEquals(FastenerState.ALIGNED, result.getState());
        assertEquals("GEOMETRY_WITHIN_THRESHOLDS", result.getReason());
    }

    @Test
    public void reversedSegmentDirectionIsStillAligned() {
        GeometryDecision result = AntiLooseGeometry.evaluate(
                FastenerTopology.NUT_PLATE, WitnessMarkRole.BRIDGES_MOVING_FIXED, true,
                p(10, 0, 0.95f), p(0, 0, 0.95f),
                p(22, 0, 0.95f), p(12, 0, 0.95f),
                100f, new GeometryThresholds(10f, 0.05f, 0.05f));

        assertEquals(FastenerState.ALIGNED, result.getState());
    }

    @Test
    public void excessiveAngleNeedsIndependentCorroboration() {
        GeometryDecision result = AntiLooseGeometry.evaluate(
                FastenerTopology.NUT_PLATE, WitnessMarkRole.BRIDGES_MOVING_FIXED, true,
                p(0, 0, 0.95f), p(10, 0, 0.95f),
                p(12, 0, 0.95f), p(12, 10, 0.95f),
                100f, new GeometryThresholds(10f, 0.20f, 0.20f));

        assertEquals(FastenerState.INSUFFICIENT, result.getState());
        assertEquals("POSSIBLE_DISPLACED_ANGLE_EXCEEDED", result.getReason());
    }

    @Test
    public void lowConfidenceIsUncertain() {
        GeometryDecision result = AntiLooseGeometry.evaluate(
                FastenerTopology.NUT_PLATE, WitnessMarkRole.BRIDGES_MOVING_FIXED, true,
                p(0, 0, 0.4f), p(10, 0, 0.95f),
                p(12, 0, 0.95f), p(22, 0, 0.95f),
                100f, new GeometryThresholds(10f, 0.05f, 0.05f));

        assertEquals(FastenerState.INSUFFICIENT, result.getState());
        assertEquals("KEYPOINT_CONFIDENCE_LOW", result.getReason());
    }

    @Test
    public void productionWithoutCalibratedThresholdsIsUncertain() {
        GeometryDecision result = AntiLooseGeometry.evaluate(
                FastenerTopology.NUT_PLATE, WitnessMarkRole.BRIDGES_MOVING_FIXED, true,
                p(0, 0, 0.95f), p(10, 0, 0.95f),
                p(12, 0, 0.95f), p(22, 0, 0.95f),
                100f, GeometryThresholds.uncalibrated());

        assertEquals(FastenerState.INSUFFICIENT, result.getState());
        assertEquals("THRESHOLDS_UNCALIBRATED", result.getReason());
    }

    @Test
    public void unknownTopologyOrOneSidedMarkIsInsufficient() {
        GeometryDecision unknown = AntiLooseGeometry.evaluate(
                FastenerTopology.UNKNOWN, WitnessMarkRole.BRIDGES_MOVING_FIXED, true,
                p(0, 0, 0.95f), p(10, 0, 0.95f),
                p(12, 0, 0.95f), p(22, 0, 0.95f),
                100f, new GeometryThresholds(10f, 0.05f, 0.05f));
        GeometryDecision oneSided = AntiLooseGeometry.evaluate(
                FastenerTopology.NUT_PLATE, WitnessMarkRole.MOVING_ONLY, true,
                p(0, 0, 0.95f), p(10, 0, 0.95f),
                p(12, 0, 0.95f), p(22, 0, 0.95f),
                100f, new GeometryThresholds(10f, 0.05f, 0.05f));

        assertEquals(FastenerState.INSUFFICIENT, unknown.getState());
        assertEquals("TOPOLOGY_UNKNOWN", unknown.getReason());
        assertEquals(FastenerState.INSUFFICIENT, oneSided.getState());
        assertEquals("MARK_DOES_NOT_BRIDGE_MOVING_FIXED", oneSided.getReason());
    }

    @Test
    public void nonFiniteKeypointEvidenceFailsClosed() {
        GeometryDecision badConfidence = AntiLooseGeometry.evaluate(
                FastenerTopology.NUT_PLATE, WitnessMarkRole.BRIDGES_MOVING_FIXED, true,
                p(0, 0, Float.NaN), p(10, 0, 0.95f),
                p(12, 0, 0.95f), p(22, 0, 0.95f),
                100f, new GeometryThresholds(10f, 0.05f, 0.05f));
        GeometryDecision badCoordinate = AntiLooseGeometry.evaluate(
                FastenerTopology.NUT_PLATE, WitnessMarkRole.BRIDGES_MOVING_FIXED, true,
                p(Float.NaN, 0, 0.95f), p(10, 0, 0.95f),
                p(12, 0, 0.95f), p(22, 0, 0.95f),
                100f, new GeometryThresholds(10f, 0.05f, 0.05f));

        assertEquals(FastenerState.INSUFFICIENT, badConfidence.getState());
        assertEquals("KEYPOINT_VALUE_INVALID", badConfidence.getReason());
        assertEquals(FastenerState.INSUFFICIENT, badCoordinate.getState());
        assertEquals("KEYPOINT_VALUE_INVALID", badCoordinate.getReason());
    }

    @Test(expected = IllegalArgumentException.class)
    public void calibratedThresholdsRejectNaN() {
        new GeometryThresholds(Float.NaN, 0.05f, 0.05f);
    }

    @Test
    public void finiteExtremeCoordinatesCannotOverflowIntoAligned() {
        GeometryDecision result = AntiLooseGeometry.evaluate(
                FastenerTopology.NUT_PLATE, WitnessMarkRole.BRIDGES_MOVING_FIXED, true,
                p(-Float.MAX_VALUE, 0, 0.95f), p(Float.MAX_VALUE, 0, 0.95f),
                p(0, -Float.MAX_VALUE, 0.95f), p(0, Float.MAX_VALUE, 0.95f),
                100f, new GeometryThresholds(10f, 0.05f, 0.05f));

        assertEquals(FastenerState.INSUFFICIENT, result.getState());
        assertEquals("POSSIBLE_DISPLACED_ANGLE_EXCEEDED", result.getReason());
    }

    @Test
    public void topologyWithoutSpecificSolverFailsClosed() {
        GeometryDecision result = AntiLooseGeometry.evaluate(
                FastenerTopology.FITTING_PIPE, WitnessMarkRole.BRIDGES_MOVING_FIXED, true,
                p(0, 0, 0.95f), p(10, 0, 0.95f),
                p(12, 0, 0.95f), p(22, 0, 0.95f),
                100f, new GeometryThresholds(10f, 0.05f, 0.05f));

        assertEquals(FastenerState.INSUFFICIENT, result.getState());
        assertEquals("TOPOLOGY_SOLVER_UNAVAILABLE", result.getReason());
    }
}
