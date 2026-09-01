package com.ar.glass.vision.realtime;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Rect;

import java.io.Closeable;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

final class FastenerInputWorkspace implements Closeable {
    private static final int LETTERBOX_COLOR = 0xFF727272;

    private final int inputSize;
    private final int pixelCount;
    private final FloatBuffer inputBuffer;
    private final int[] pixels;
    private final Bitmap letterboxBitmap;
    private final Canvas letterboxCanvas;
    private final Paint bitmapPaint;
    private final Rect destinationRect = new Rect();

    FastenerInputWorkspace(int inputSize) {
        this.inputSize = inputSize;
        pixelCount = inputSize * inputSize;
        inputBuffer = ByteBuffer
                .allocateDirect(pixelCount * 3 * Float.BYTES)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        pixels = new int[pixelCount];
        letterboxBitmap = Bitmap.createBitmap(
                inputSize, inputSize, Bitmap.Config.ARGB_8888);
        letterboxCanvas = new Canvas(letterboxBitmap);
        bitmapPaint = new Paint(Paint.FILTER_BITMAP_FLAG);
    }

    FloatBuffer prepare(Bitmap source, LetterboxTransform transform) {
        letterboxCanvas.drawColor(LETTERBOX_COLOR);
        destinationRect.set(
                transform.getPadLeft(),
                transform.getPadTop(),
                transform.getPadLeft() + transform.getResizedWidth(),
                transform.getPadTop() + transform.getResizedHeight());
        letterboxCanvas.drawBitmap(source, null, destinationRect, bitmapPaint);
        letterboxBitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize);

        inputBuffer.clear();
        for (int index = 0; index < pixelCount; index++) {
            int pixel = pixels[index];
            inputBuffer.put(index, ((pixel >> 16) & 0xFF) / 255f);
            inputBuffer.put(pixelCount + index, ((pixel >> 8) & 0xFF) / 255f);
            inputBuffer.put(2 * pixelCount + index, (pixel & 0xFF) / 255f);
        }
        inputBuffer.position(0);
        return inputBuffer;
    }

    @Override
    public void close() {
        if (!letterboxBitmap.isRecycled()) {
            letterboxBitmap.recycle();
        }
    }
}
