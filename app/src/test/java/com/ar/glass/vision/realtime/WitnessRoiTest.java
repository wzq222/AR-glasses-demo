package com.ar.glass.vision.realtime;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class WitnessRoiTest {
    @Test
    public void expandsCenteredDetectionToTrainingScale() {
        WitnessRoi roi = WitnessRoi.fromDetection(
                new Detection(80f, 40f, 120f, 80f, 0.9f, 0),
                300,
                200);

        assertEquals(70, roi.getLeft());
        assertEquals(30, roi.getTop());
        assertEquals(130, roi.getRight());
        assertEquals(90, roi.getBottom());
        assertEquals(60, roi.getWidth());
        assertEquals(60, roi.getHeight());
    }

    @Test
    public void independentlyClampsTrainingCropAtFrameEdgesWithoutShifting() {
        WitnessRoi roi = WitnessRoi.fromDetection(
                new Detection(-4f, 176f, 36f, 210f, 0.9f, 0),
                300,
                200);

        assertEquals(0, roi.getLeft());
        assertEquals(163, roi.getTop());
        assertEquals(46, roi.getRight());
        assertEquals(200, roi.getBottom());
        assertEquals(46, roi.getWidth());
        assertEquals(37, roi.getHeight());
    }

    @Test
    public void independentlyClampsOversizedTrainingCropToWholeFrame() {
        WitnessRoi roi = WitnessRoi.fromDetection(
                new Detection(0f, 0f, 300f, 190f, 0.9f, 0),
                300,
                200);

        assertEquals(0, roi.getLeft());
        assertEquals(0, roi.getTop());
        assertEquals(300, roi.getRight());
        assertEquals(200, roi.getBottom());
    }

    @Test
    public void enforcesTrainingMinimumSideBeforeIndependentClamp() {
        WitnessRoi roi = WitnessRoi.fromDetection(
                new Detection(10f, 10f, 14f, 12f, 0.9f, 0),
                300,
                200);

        assertEquals(4, roi.getLeft());
        assertEquals(3, roi.getTop());
        assertEquals(20, roi.getRight());
        assertEquals(19, roi.getBottom());
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsDetectionThatDoesNotIntersectTheFrame() {
        WitnessRoi.fromDetection(
                new Detection(310f, 20f, 350f, 60f, 0.9f, 0),
                300,
                200);
    }
}
