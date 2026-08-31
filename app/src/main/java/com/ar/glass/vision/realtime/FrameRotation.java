package com.ar.glass.vision.realtime;

import java.util.Arrays;

public final class FrameRotation {
    private FrameRotation() {
    }

    public static RotatedFrame rotate(int[] source, int width, int height, int clockwiseDegrees) {
        int outputWidth = clockwiseDegrees == 90 || clockwiseDegrees == 270
                ? height : width;
        int outputHeight = clockwiseDegrees == 90 || clockwiseDegrees == 270
                ? width : height;
        int[] output = new int[checkedPixelCount(width, height)];
        rotateInto(source, width, height, clockwiseDegrees, output);
        return new RotatedFrame(output, outputWidth, outputHeight);
    }

    public static void rotateInto(
            int[] source,
            int width,
            int height,
            int clockwiseDegrees,
            int[] destination) {
        int pixelCount = checkedPixelCount(width, height);
        if (source == null || source.length < pixelCount
                || destination == null || destination.length < pixelCount) {
            throw new IllegalArgumentException("pixel buffers must contain the full frame");
        }
        if (clockwiseDegrees != 0 && clockwiseDegrees != 90
                && clockwiseDegrees != 180 && clockwiseDegrees != 270) {
            throw new IllegalArgumentException("rotation must be 0, 90, 180, or 270 degrees");
        }
        if (source == destination && clockwiseDegrees != 0) {
            throw new IllegalArgumentException("rotated output requires a distinct destination");
        }
        int outputWidth = clockwiseDegrees == 90 || clockwiseDegrees == 270
                ? height : width;

        if (clockwiseDegrees == 0) {
            System.arraycopy(source, 0, destination, 0, pixelCount);
        } else {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int targetX;
                    int targetY;
                    if (clockwiseDegrees == 90) {
                        targetX = height - 1 - y;
                        targetY = x;
                    } else if (clockwiseDegrees == 180) {
                        targetX = width - 1 - x;
                        targetY = height - 1 - y;
                    } else {
                        targetX = y;
                        targetY = width - 1 - x;
                    }
                    destination[targetY * outputWidth + targetX] = source[y * width + x];
                }
            }
        }
    }

    private static int checkedPixelCount(int width, int height) {
        if (width <= 0 || height <= 0 || width > Integer.MAX_VALUE / height) {
            throw new IllegalArgumentException("frame dimensions must be positive and bounded");
        }
        return width * height;
    }

    public static final class RotatedFrame {
        private final int[] pixels;
        private final int width;
        private final int height;

        private RotatedFrame(int[] pixels, int width, int height) {
            this.pixels = pixels;
            this.width = width;
            this.height = height;
        }

        public int[] copyPixels() { return Arrays.copyOf(pixels, pixels.length); }
        public int getWidth() { return width; }
        public int getHeight() { return height; }
    }
}
