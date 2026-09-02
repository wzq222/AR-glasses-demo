package com.ar.glass.vision.realtime;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class WitnessRoiTest {
    @Test public void expandsDetectionAndClampsAtFrameBoundary() {
        Detection detection = new Detection(0f, 10f, 100f, 70f, 0.8f, 0);
        WitnessRoi roi = WitnessRoi.fromDetection(detection, 300, 200);
        assertEquals(0, roi.getLeft());
        assertEquals(0, roi.getTop());
        assertEquals(125, roi.getRight());
        assertEquals(115, roi.getBottom());
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsDetectionOutsideFrame() {
        WitnessRoi.fromDetection(new Detection(400f, 10f, 450f, 60f, 0.8f, 0), 300, 200);
    }
}
