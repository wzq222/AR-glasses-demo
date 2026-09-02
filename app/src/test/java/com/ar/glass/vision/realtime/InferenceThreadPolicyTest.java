package com.ar.glass.vision.realtime;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class InferenceThreadPolicyTest {
    @Test
    public void boundsInferenceToTheFourPerformanceCores() {
        assertEquals(4, InferenceThreadPolicy.intraOpThreads());
    }
}
