package com.ar.glass.vision.realtime;

public final class FrameCropper {
    private FrameCropper() {
    }

    public static void cropInto(
            int[] source,
            int sourceWidth,
            int sourceHeight,
            int left,
            int top,
            int right,
            int bottom,
            int[] destination) {
        if (source == null || sourceWidth <= 0 || sourceHeight <= 0
                || source.length < sourceWidth * sourceHeight) {
            throw new IllegalArgumentException("source frame is invalid");
        }
        if (left < 0 || top < 0 || right > sourceWidth || bottom > sourceHeight
                || right <= left || bottom <= top) {
            throw new IllegalArgumentException("crop rectangle is outside the source frame");
        }
        int cropWidth = right - left;
        int cropHeight = bottom - top;
        int cropPixels = cropWidth * cropHeight;
        if (destination == null || destination.length < cropPixels) {
            throw new IllegalArgumentException("destination is shorter than the crop");
        }
        for (int row = 0; row < cropHeight; row++) {
            System.arraycopy(
                    source,
                    (top + row) * sourceWidth + left,
                    destination,
                    row * cropWidth,
                    cropWidth);
        }
    }
}
