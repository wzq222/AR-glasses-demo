package com.ar.glass.vision.realtime;

import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class InferenceGateTest {
    @Test
    public void allowsOnlyOneInferenceAndCoolsDownAfterCompletion() {
        InferenceGate gate = new InferenceGate(1_000L);

        assertTrue(gate.tryAcquire(1_000L));
        assertFalse(gate.tryAcquire(2_000L));

        gate.release(3_000L);
        assertFalse(gate.tryAcquire(3_999L));
        assertTrue(gate.tryAcquire(4_000L));
    }
}
