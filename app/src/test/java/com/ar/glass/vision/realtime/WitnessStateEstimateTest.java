package com.ar.glass.vision.realtime;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class WitnessStateEstimateTest {
    private static final float[] POINTS = {0.2f, 0.5f, 0.4f, 0.5f, 0.4f, 0.5f, 0.7f, 0.5f};

    @Test public void routesPointAnglesToHumanReviewBuckets() {
        assertEquals(WitnessTriage.LIKELY_ALIGNED,
                WitnessStateEstimate.measured(3f, POINTS, 5.0).getTriage());
        assertEquals(WitnessTriage.POSSIBLE_DISPLACED,
                WitnessStateEstimate.measured(8f, POINTS, 5.0).getTriage());
        assertEquals(WitnessTriage.HIGH_SUSPICION,
                WitnessStateEstimate.measured(15f, POINTS, 5.0).getTriage());
    }

    @Test public void measuredResultCarriesConservativeInterval() {
        WitnessStateEstimate result = WitnessStateEstimate.measured(8f, POINTS, 7.5);
        assertTrue(result.isMeasured());
        assertEquals(1.7f, result.getLowerDegrees(), 0.0001f);
        assertEquals(14.3f, result.getUpperDegrees(), 0.0001f);
        assertEquals(7.5, result.getInferenceMillis(), 0.0001);
    }

    @Test public void insufficientResultNeverLooksMeasured() {
        WitnessStateEstimate result = WitnessStateEstimate.insufficient("LOW_RESOLUTION");
        assertFalse(result.isMeasured());
        assertEquals(WitnessTriage.INSUFFICIENT, result.getTriage());
        assertEquals("LOW_RESOLUTION", result.getReason());
    }
}
