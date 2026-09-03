package com.ar.glass.sop;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class SopStepPolicyTest {
    @Test public void normalDecisionIsSuccessfulAndAuditable() {
        assertEquals("succeeded", SopStepPolicy.statusForDecision(0));
        assertEquals("confirmed_ok", SopStepPolicy.humanDecision(0));
    }

    @Test public void suspectedAndUnableRemainUncertain() {
        assertEquals("uncertain", SopStepPolicy.statusForDecision(1));
        assertEquals("suspected", SopStepPolicy.humanDecision(1));
        assertEquals("uncertain", SopStepPolicy.statusForDecision(2));
        assertEquals("unable_to_judge", SopStepPolicy.humanDecision(2));
    }

    @Test public void idempotencyKeyIsStableAcrossRetries() {
        assertEquals("run-1:FASTENER_CHECK",
                SopStepPolicy.idempotencyKey("run-1", "FASTENER_CHECK"));
    }
}
