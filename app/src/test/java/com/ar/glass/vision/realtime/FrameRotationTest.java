package com.ar.glass.vision.realtime;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class FrameRotationTest {
    private static final int[] SOURCE = {1, 2, 3, 4, 5, 6};

    @Test
    public void zeroDegreesPreservesDimensionsAndPixels() {
        FrameRotation.RotatedFrame frame = FrameRotation.rotate(SOURCE, 2, 3, 0);

        assertEquals(2, frame.getWidth());
        assertEquals(3, frame.getHeight());
        assertArrayEquals(new int[] {1, 2, 3, 4, 5, 6}, frame.copyPixels());
    }

    @Test
    public void ninetyDegreesRotatesClockwiseAndSwapsDimensions() {
        FrameRotation.RotatedFrame frame = FrameRotation.rotate(SOURCE, 2, 3, 90);

        assertEquals(3, frame.getWidth());
        assertEquals(2, frame.getHeight());
        assertArrayEquals(new int[] {5, 3, 1, 6, 4, 2}, frame.copyPixels());
    }

    @Test
    public void oneHundredEightyDegreesRotatesPixels() {
        FrameRotation.RotatedFrame frame = FrameRotation.rotate(SOURCE, 2, 3, 180);

        assertEquals(2, frame.getWidth());
        assertEquals(3, frame.getHeight());
        assertArrayEquals(new int[] {6, 5, 4, 3, 2, 1}, frame.copyPixels());
    }

    @Test
    public void twoHundredSeventyDegreesRotatesClockwiseAndSwapsDimensions() {
        FrameRotation.RotatedFrame frame = FrameRotation.rotate(SOURCE, 2, 3, 270);

        assertEquals(3, frame.getWidth());
        assertEquals(2, frame.getHeight());
        assertArrayEquals(new int[] {2, 4, 6, 1, 3, 5}, frame.copyPixels());
    }

    @Test
    public void rotateIntoWritesIntoReusableDestinationWithoutAllocatingAResultArray() {
        int[] destination = {-1, -1, -1, -1, -1, -1, 12345};

        FrameRotation.rotateInto(SOURCE, 2, 3, 90, destination);

        assertArrayEquals(new int[] {5, 3, 1, 6, 4, 2, 12345}, destination);
    }
}
