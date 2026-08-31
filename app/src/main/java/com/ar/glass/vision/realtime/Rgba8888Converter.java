package com.ar.glass.vision.realtime;

public final class Rgba8888Converter {
    private static final int RGBA_BYTES = 4;

    private Rgba8888Converter() {
    }

    public static int[] toArgb(
            byte[] plane,
            int width,
            int height,
            int rowStride,
            int pixelStride) {
        int[] pixels = new int[checkedPixelCount(width, height)];
        toArgb(plane, width, height, rowStride, pixelStride, pixels);
        return pixels;
    }

    public static void toArgb(
            byte[] plane,
            int width,
            int height,
            int rowStride,
            int pixelStride,
            int[] destination) {
        if (plane == null) {
            throw new IllegalArgumentException("plane must not be null");
        }
        int pixelCount = checkedPixelCount(width, height);
        if (destination == null || destination.length < pixelCount) {
            throw new IllegalArgumentException("destination is shorter than the frame");
        }
        if (pixelStride < RGBA_BYTES) {
            throw new IllegalArgumentException("invalid RGBA pixel stride");
        }
        long minimumRowBytes = (long) (width - 1) * pixelStride + RGBA_BYTES;
        if (rowStride < minimumRowBytes) {
            throw new IllegalArgumentException("invalid RGBA plane dimensions or strides");
        }
        long requiredBytes = (long) (height - 1) * rowStride
                + (long) (width - 1) * pixelStride + RGBA_BYTES;
        if (requiredBytes > plane.length) {
            throw new IllegalArgumentException("RGBA plane is shorter than its strides require");
        }

        for (int y = 0; y < height; y++) {
            int rowOffset = y * rowStride;
            for (int x = 0; x < width; x++) {
                int sourceOffset = rowOffset + x * pixelStride;
                int alpha = plane[sourceOffset] & 0xFF;
                int red = plane[sourceOffset + 1] & 0xFF;
                int green = plane[sourceOffset + 2] & 0xFF;
                int blue = plane[sourceOffset + 3] & 0xFF;
                destination[y * width + x] = (alpha << 24)
                        | (red << 16)
                        | (green << 8)
                        | blue;
            }
        }
    }

    private static int checkedPixelCount(int width, int height) {
        if (width <= 0 || height <= 0 || width > Integer.MAX_VALUE / height) {
            throw new IllegalArgumentException("frame dimensions must be positive and bounded");
        }
        return width * height;
    }
}
