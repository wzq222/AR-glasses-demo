package com.ar.glass.vision.realtime;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class SquareRoiTest {
    @Test
    public void expandsCenteredDetectionToTrainingScaleSquare() {
        SquareRoi roi = SquareRoi.fromDetection(
                new Detection(80f, 40f, 120f, 80f, 0.9f, 0),
                300,
                200);

        assertEquals(70, roi.getLeft());
        assertEquals(30, roi.getTop());
        assertEquals(60, roi.getSide());
        assertEquals(130, roi.getRight());
        assertEquals(90, roi.getBottom());
    }

    @Test
    public void shiftsExpandedSquareInsideImageAtEdges() {
        SquareRoi roi = SquareRoi.fromDetection(
                new Detection(-4f, 176f, 36f, 210f, 0.9f, 0),
                300,
                200);

        assertEquals(0, roi.getLeft());
        assertEquals(140, roi.getTop());
        assertEquals(60, roi.getSide());
        assertEquals(60, roi.getRight());
        assertEquals(200, roi.getBottom());
    }

    @Test
    public void clampsLargeDetectionToShortImageDimension() {
        SquareRoi roi = SquareRoi.fromDetection(
                new Detection(0f, 0f, 300f, 190f, 0.9f, 0),
                300,
                200);

        assertEquals(50, roi.getLeft());
        assertEquals(0, roi.getTop());
        assertEquals(200, roi.getSide());
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsDetectionThatDoesNotIntersectTheFrame() {
        SquareRoi.fromDetection(
                new Detection(310f, 20f, 350f, 60f, 0.9f, 0),
                300,
                200);
    }
}
