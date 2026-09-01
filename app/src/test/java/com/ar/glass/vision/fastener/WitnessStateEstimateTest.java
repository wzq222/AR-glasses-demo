package com.ar.glass.vision.fastener;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class WitnessStateEstimateTest {
    @Test
    public void exactThreeDegreesUsesLikelyAlignedPointBucketDespiteUncertainty() {
        WitnessStateEstimate result = WitnessStateEstimate.experimental(3f, 12.5);

        assertEquals(3f, result.getAngle().getPointEstimateDegrees(), 0f);
        assertEquals(0f, result.getAngle().getLowerDegrees(), 0f);
        assertEquals(9.3f, result.getAngle().getUpperDegrees(), 0.0001f);
        assertEquals(WitnessReviewHint.LIKELY_ALIGNED, result.getReviewHint());
        assertEquals("POINT_ANGLE_AT_OR_BELOW_REVIEW_THRESHOLD", result.getReviewReason());
        assertEquals(12.5, result.getInferenceMillis(), 0f);
    }

    @Test
    public void immediatelyAboveThreeDegreesUsesReviewBucket() {
        WitnessStateEstimate result = WitnessStateEstimate.experimental(3.0001f, 4.0);

        assertEquals(WitnessReviewHint.POSSIBLE_DISPLACED, result.getReviewHint());
        assertEquals("POINT_ANGLE_REVIEW_REQUIRED", result.getReviewReason());
    }

    @Test
    public void immediatelyBelowFifteenDegreesUsesReviewBucket() {
        WitnessStateEstimate result = WitnessStateEstimate.experimental(14.9999f, 4.0);

        assertEquals(WitnessReviewHint.POSSIBLE_DISPLACED, result.getReviewHint());
        assertEquals("POINT_ANGLE_REVIEW_REQUIRED", result.getReviewReason());
    }

    @Test
    public void exactFifteenDegreesRequiresSecondView() {
        WitnessStateEstimate result = WitnessStateEstimate.experimental(15f, 4.0);

        assertEquals(WitnessReviewHint.POSSIBLE_DISPLACED, result.getReviewHint());
        assertEquals("POINT_ANGLE_SECOND_VIEW_REQUIRED", result.getReviewReason());
        assertEquals(8.7f, result.getAngle().getLowerDegrees(), 0.0001f);
        assertEquals(21.3f, result.getAngle().getUpperDegrees(), 0.0001f);
    }

    @Test
    public void uncertaintyClampsAtNinetyDegrees() {
        WitnessStateEstimate result = WitnessStateEstimate.experimental(89f, 4.0);

        assertEquals(82.7f, result.getAngle().getLowerDegrees(), 0.0001f);
        assertEquals(90f, result.getAngle().getUpperDegrees(), 0f);
    }
}
