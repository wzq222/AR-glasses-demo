package com.ar.glass.vision.realtime;

import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class InferenceGateTest {
    @Test
    public void allowsOnlyOneInferenceAndThrottlesStartsForFiveHundredMilliseconds() {
        InferenceGate gate = new InferenceGate(500L);

        assertTrue(gate.tryAcquire(1_000L));
        assertFalse(gate.tryAcquire(2_000L));

        gate.release();
        assertFalse(gate.tryAcquire(1_499L));
        assertTrue(gate.tryAcquire(1_500L));
    }
}
