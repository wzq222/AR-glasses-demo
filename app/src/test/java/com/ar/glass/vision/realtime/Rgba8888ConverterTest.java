package com.ar.glass.vision.realtime;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class Rgba8888ConverterTest {
    @Test
    public void convertsCameraXArgbPixelsWhileSkippingRowPadding() {
        byte[] plane = new byte[] {
                (byte) 0x44, (byte) 0x11, (byte) 0x22, (byte) 0x33,
                (byte) 0xDD, (byte) 0xAA, (byte) 0xBB, (byte) 0xCC,
                9, 9, 9, 9,
                (byte) 0x04, (byte) 0x01, (byte) 0x02, (byte) 0x03,
                (byte) 0x98, (byte) 0xFE, (byte) 0xDC, (byte) 0xBA,
                8, 8, 8, 8
        };

        int[] pixels = Rgba8888Converter.toArgb(
                plane, 2, 2, 12, 4);

        assertArrayEquals(new int[] {
                0x44112233,
                0xDDAABBCC,
                0x04010203,
                0x98FEDCBA
        }, pixels);
    }

    @Test
    public void honorsPixelStrideLargerThanFourBytes() {
        byte[] plane = new byte[] {
                4, 1, 2, 3, 99, 99,
                8, 5, 6, 7
        };

        int[] pixels = Rgba8888Converter.toArgb(
                plane, 2, 1, 10, 6);

        assertArrayEquals(new int[] {0x04010203, 0x08050607}, pixels);
    }

    @Test
    public void writesIntoReusableDestinationWithoutAllocatingAResultArray() {
        byte[] plane = new byte[] {
                4, 1, 2, 3,
                8, 5, 6, 7
        };
        int[] destination = new int[] {-1, -1, 12345};

        Rgba8888Converter.toArgb(
                plane, 2, 1, 8, 4, destination);

        assertArrayEquals(new int[] {0x04010203, 0x08050607, 12345}, destination);
    }
}
