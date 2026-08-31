package com.ar.glass.vision.realtime;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class FrameCropperTest {
    @Test
    public void cropsTheSharedViewportBeforeRotation() {
        int[] source = new int[] {
                0, 1, 2, 3,
                4, 5, 6, 7,
                8, 9, 10, 11
        };
        int[] destination = new int[6];

        FrameCropper.cropInto(source, 4, 3, 1, 1, 4, 3, destination);

        assertArrayEquals(new int[] {5, 6, 7, 9, 10, 11}, destination);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsCropOutsideTheSourceFrame() {
        FrameCropper.cropInto(new int[12], 4, 3, -1, 0, 4, 3, new int[12]);
    }
}
