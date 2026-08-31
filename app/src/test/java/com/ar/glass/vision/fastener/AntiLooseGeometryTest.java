package com.ar.glass.vision.fastener;

import com.google.gson.Gson;

import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class AntiLooseGeometryTest {
    private static final class TriageVectors {
        float review_degrees;
        float high_suspicion_degrees;
        TriageCase[] cases;
    }

    private static final class TriageCase {
        String id;
        float point;
        float lower;
        float upper;
        boolean second_view;
        String hint;
        String reason;
    }

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
        assertEquals(WitnessReviewHint.LIKELY_ALIGNED, result.getReviewHint());
        assertEquals(0f, result.getAngleDegrees(), 0.0001f);
        assertEquals(0f, result.getAngleLowerDegrees(), 0.0001f);
        assertEquals(0f, result.getAngleUpperDegrees(), 0.0001f);
    }

    @Test
    public void provisionalTriageRoutesExactThreeDegreeUpperBoundaryAsLikelyAligned() {
        GeometryDecision result = AntiLooseGeometry.triageAngle(
                new AngleInterval(2.5f, 2f, 3f),
                ProvisionalTriageThresholds.defaults(),
                false);

        assertEquals(FastenerState.INSUFFICIENT, result.getState());
        assertEquals(WitnessReviewHint.LIKELY_ALIGNED, result.getReviewHint());
        assertEquals("ANGLE_INTERVAL_BELOW_REVIEW_THRESHOLD", result.getReason());
    }

    @Test
    public void provisionalTriageRoutesThreeDegreeCrossingToReview() {
        GeometryDecision result = AntiLooseGeometry.triageAngle(
                new AngleInterval(3f, 2.5f, 3.5f),
                ProvisionalTriageThresholds.defaults(),
                false);

        assertEquals(WitnessReviewHint.POSSIBLE_DISPLACED, result.getReviewHint());
        assertEquals("ANGLE_INTERVAL_CROSSES_REVIEW_THRESHOLD", result.getReason());
    }

    @Test
    public void provisionalTriageRequiresSecondViewAtExactFifteenDegrees() {
        AngleInterval interval = new AngleInterval(18f, 15f, 21f);
        GeometryDecision unconfirmed = AntiLooseGeometry.triageAngle(
                interval, ProvisionalTriageThresholds.defaults(), false);
        GeometryDecision confirmed = AntiLooseGeometry.triageAngle(
                interval, ProvisionalTriageThresholds.defaults(), true);

        assertEquals(WitnessReviewHint.POSSIBLE_DISPLACED, unconfirmed.getReviewHint());
        assertEquals("SECOND_VIEW_CONFIRMATION_REQUIRED", unconfirmed.getReason());
        assertEquals(WitnessReviewHint.LIKELY_DISPLACED, confirmed.getReviewHint());
        assertEquals("ANGLE_INTERVAL_HIGH_SUSPICION_CONFIRMED", confirmed.getReason());
        assertEquals(FastenerState.INSUFFICIENT, confirmed.getState());
    }

    @Test
    public void uncalibratedEvaluationUsesProvidedConfidenceInterval() {
        GeometryDecision result = AntiLooseGeometry.evaluate(
                FastenerTopology.NUT_PLATE, WitnessMarkRole.BRIDGES_MOVING_FIXED, true,
                p(0, 0, 0.95f), p(10, 0, 0.95f),
                p(12, 0, 0.95f), p(12, 10, 0.95f),
                100f, GeometryThresholds.uncalibrated(),
                new AngleInterval(20f, 16f, 24f), true);

        assertEquals(FastenerState.INSUFFICIENT, result.getState());
        assertEquals(WitnessReviewHint.LIKELY_DISPLACED, result.getReviewHint());
        assertEquals(16f, result.getAngleLowerDegrees(), 0.0001f);
        assertEquals(24f, result.getAngleUpperDegrees(), 0.0001f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void angleIntervalRejectsInvertedBounds() {
        new AngleInterval(4f, 5f, 3f);
    }

    @Test
    public void sharedPythonJavaTriageVectorsMatch() {
        InputStreamReader reader = new InputStreamReader(
                AntiLooseGeometryTest.class.getResourceAsStream("/witness-triage-v1.json"),
                StandardCharsets.UTF_8);
        TriageVectors vectors = new Gson().fromJson(reader, TriageVectors.class);
        ProvisionalTriageThresholds thresholds = new ProvisionalTriageThresholds(
                vectors.review_degrees, vectors.high_suspicion_degrees);

        for (TriageCase item : vectors.cases) {
            GeometryDecision result = AntiLooseGeometry.triageAngle(
                    new AngleInterval(item.point, item.lower, item.upper),
                    thresholds,
                    item.second_view);
            assertEquals(item.id, WitnessReviewHint.valueOf(item.hint), result.getReviewHint());
            assertEquals(item.id, item.reason, result.getReason());
            assertEquals(item.id, FastenerState.INSUFFICIENT, result.getState());
        }
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
