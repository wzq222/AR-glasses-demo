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
        if (plane == null) {
            throw new IllegalArgumentException("plane must not be null");
        }
        int minimumRowBytes = (width - 1) * pixelStride + RGBA_BYTES;
        if (width <= 0 || height <= 0 || pixelStride < RGBA_BYTES
                || rowStride < minimumRowBytes) {
            throw new IllegalArgumentException("invalid RGBA plane dimensions or strides");
        }
        long requiredBytes = (long) (height - 1) * rowStride
                + (long) (width - 1) * pixelStride + RGBA_BYTES;
        if (requiredBytes > plane.length) {
            throw new IllegalArgumentException("RGBA plane is shorter than its strides require");
        }

        int[] pixels = new int[width * height];
        for (int y = 0; y < height; y++) {
            int rowOffset = y * rowStride;
            for (int x = 0; x < width; x++) {
                int sourceOffset = rowOffset + x * pixelStride;
                int red = plane[sourceOffset] & 0xFF;
                int green = plane[sourceOffset + 1] & 0xFF;
                int blue = plane[sourceOffset + 2] & 0xFF;
                int alpha = plane[sourceOffset + 3] & 0xFF;
                pixels[y * width + x] = (alpha << 24)
                        | (red << 16)
                        | (green << 8)
                        | blue;
            }
        }
        return pixels;
    }
}
