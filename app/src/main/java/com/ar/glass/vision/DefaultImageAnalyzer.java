package com.ar.glass.vision;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;

import com.google.android.gms.tasks.Tasks;
import com.google.mlkit.vision.barcode.BarcodeScanner;
import com.google.mlkit.vision.barcode.BarcodeScannerOptions;
import com.google.mlkit.vision.barcode.BarcodeScanning;
import com.google.mlkit.vision.barcode.common.Barcode;
import com.google.mlkit.vision.common.InputImage;

import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * 识别接口的实现。
 *
 * 二维码识别已接入 ML Kit Barcode Scanning（离线，对模糊/透视/旋转鲁棒）。
 * 针对「远距离小二维码」「模糊二维码」识别率低的问题，在单次识别的基础上
 * 增加多级策略：对比度增强 + 放大 2 倍，任一策略命中即返回。
 *
 * 防松线错位、电压表数字识别尚未接入真实算法：
 * - 防松线错位：自定义图像处理 / 目标检测模型
 * - 电压表数字：ML Kit Text Recognition 或 PaddleOCR
 */
public class DefaultImageAnalyzer implements ImageAnalyzer {

    /** 放大识别时限制的最长边，避免超大图导致 OOM */
    private static final int MAX_DIMENSION = 2000;

    private final BarcodeScanner scanner = BarcodeScanning.getClient(
            new BarcodeScannerOptions.Builder()
                    .setBarcodeFormats(Barcode.FORMAT_QR_CODE)
                    .build());

    @Override
    public String decodeQrCode(Bitmap bitmap) {
        if (bitmap == null) return null;

        Bitmap src = toArgb8888(bitmap);
        try {
            // 1. 原图识别
            String result = tryDecode(src);
            if (result != null) return result;

            // 2. 对比度增强识别（缓解模糊、低对比度）
            Bitmap enhanced = enhanceContrast(src);
            result = tryDecode(enhanced);
            enhanced.recycle();
            if (result != null) return result;

            // 3. 放大 2 倍识别（缓解远距离、小二维码）
            result = tryDecodeScaled(src);
            if (result != null) return result;

            // 4. 放大 2 倍 + 对比度增强（小 + 模糊叠加场景）
            Bitmap limited = limitDimension(src, MAX_DIMENSION);
            Bitmap scaled = Bitmap.createScaledBitmap(
                    limited, limited.getWidth() * 2, limited.getHeight() * 2, true);
            if (limited != src) limited.recycle();
            Bitmap enhanced2 = enhanceContrast(scaled);
            result = tryDecode(enhanced2);
            enhanced2.recycle();
            scaled.recycle();
            if (result != null) return result;

            return null;
        } finally {
            if (src != bitmap && src != null) {
                src.recycle();
            }
        }
    }

    /** 对原图限制尺寸后放大 2 倍再识别，避免超大图直接放大导致 OOM */
    private String tryDecodeScaled(Bitmap src) {
        Bitmap limited = limitDimension(src, MAX_DIMENSION);
        Bitmap scaled = Bitmap.createScaledBitmap(
                limited, limited.getWidth() * 2, limited.getHeight() * 2, true);
        if (limited != src) limited.recycle();
        String result = tryDecode(scaled);
        scaled.recycle();
        return result;
    }

    private String tryDecode(Bitmap bitmap) {
        try {
            InputImage image = InputImage.fromBitmap(bitmap, 0);
            List<Barcode> barcodes = Tasks.await(scanner.process(image), 8, TimeUnit.SECONDS);
            if (barcodes != null) {
                for (Barcode barcode : barcodes) {
                    if (barcode.getFormat() == Barcode.FORMAT_QR_CODE
                            && barcode.getRawValue() != null
                            && !barcode.getRawValue().isEmpty()) {
                        return barcode.getRawValue();
                    }
                }
            }
        } catch (Exception ignored) {
        }
        return null;
    }

    private Bitmap toArgb8888(Bitmap bitmap) {
        if (bitmap.getConfig() == Bitmap.Config.ARGB_8888) return bitmap;
        return bitmap.copy(Bitmap.Config.ARGB_8888, false);
    }

    private Bitmap limitDimension(Bitmap src, int maxDimension) {
        int max = Math.max(src.getWidth(), src.getHeight());
        if (max <= maxDimension) return src;
        float ratio = (float) maxDimension / max;
        return Bitmap.createScaledBitmap(
                src, Math.round(src.getWidth() * ratio), Math.round(src.getHeight() * ratio), true);
    }

    /** 提升对比度，强化模糊二维码的黑白边缘 */
    private Bitmap enhanceContrast(Bitmap src) {
        Bitmap out = Bitmap.createBitmap(src.getWidth(), src.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(out);
        Paint paint = new Paint();
        float contrast = 1.6f;
        float offset = 128f * (1f - contrast);
        ColorMatrix cm = new ColorMatrix(new float[]{
                contrast, 0, 0, 0, offset,
                0, contrast, 0, 0, offset,
                0, 0, contrast, 0, offset,
                0, 0, 0, 1, 0
        });
        paint.setColorFilter(new ColorMatrixColorFilter(cm));
        canvas.drawBitmap(src, 0, 0, paint);
        return out;
    }

    @Override
    public boolean isNutLoose(Bitmap bitmap) {
        // TODO 接入防松线错位检测
        return false;
    }

    @Override
    public String readMeterValue(Bitmap bitmap) {
        // TODO 接入电压表数字 OCR
        return null;
    }
}
