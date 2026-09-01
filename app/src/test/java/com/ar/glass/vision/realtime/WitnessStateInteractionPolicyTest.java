package com.ar.glass.vision.realtime;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class WitnessStateInteractionPolicyTest {
    @Test
    public void defaultBuildKeepsStateReviewDisabled() {
        WitnessStateInteractionPolicy policy =
                WitnessStateInteractionPolicy.forRuntime(false, false);

        assertEquals(WitnessStateInteractionPolicy.Availability.DISABLED,
                policy.getAvailability());
        assertFalse(policy.canTapCandidate());
    }

    @Test
    public void failedExperimentalEstimatorIsUnavailableAndNotTappable() {
        WitnessStateInteractionPolicy policy =
                WitnessStateInteractionPolicy.forRuntime(true, false);

        assertEquals(WitnessStateInteractionPolicy.Availability.UNAVAILABLE,
                policy.getAvailability());
        assertFalse(policy.canTapCandidate());
    }

    @Test
    public void experimentalCandidateIsTappableOnlyWhenEstimatorIsReady() {
        WitnessStateInteractionPolicy policy =
                WitnessStateInteractionPolicy.forRuntime(true, true);

        assertEquals(WitnessStateInteractionPolicy.Availability.READY,
                policy.getAvailability());
        assertTrue(policy.canTapCandidate());
    }
}
