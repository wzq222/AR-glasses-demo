package com.ar.glass.ui;

import android.content.ContentValues;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import com.ar.glass.R;
import com.ar.glass.vision.MeterReading;
import com.ar.glass.vision.Vision;
import com.ar.glass.vision.YoloDetector;
import com.ar.glass.vision.YoloDetectorHolder;
import com.ar.glass.vision.cloud.MeterCloudOcr;

import java.io.File;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * 全屏图片查看器
 * 长按图片（或点底部“检测”按钮）弹出菜单：
 * 保存到相册 / 二维码识别 / YOLO 目标检测 / 万用表读数识别
 */
public class ImageViewerActivity extends AppCompatActivity {

    private static final int DETECT_QR = 0;
    private static final int DETECT_YOLO = 1;
    private static final int DETECT_METER = 2;

    private ImageView ivFullImage;
    private TextView tvViewerTitle;
    private TextView tvViewerIndex;
    private Button btnViewerBack;
    private Button btnPrev;
    private Button btnNext;
    private Button btnRecognize;

    private ArrayList<String> imagePaths;
    private int currentIndex = 0;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private Bitmap mAnnotatedBitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_viewer);

        ivFullImage = findViewById(R.id.ivFullImage);
        tvViewerTitle = findViewById(R.id.tvViewerTitle);
        tvViewerIndex = findViewById(R.id.tvViewerIndex);
        btnViewerBack = findViewById(R.id.btnViewerBack);
        btnPrev = findViewById(R.id.btnPrev);
        btnNext = findViewById(R.id.btnNext);
        btnRecognize = findViewById(R.id.btnRecognize);

        imagePaths = getIntent().getStringArrayListExtra("image_paths");
        currentIndex = getIntent().getIntExtra("start_index", 0);
        String title = getIntent().getStringExtra("title");
        if (title != null) tvViewerTitle.setText(title);

        if (imagePaths == null || imagePaths.isEmpty()) {
            finish();
            return;
        }

        btnViewerBack.setOnClickListener(v -> finish());
        btnPrev.setOnClickListener(v -> showImage(currentIndex - 1));
        btnNext.setOnClickListener(v -> showImage(currentIndex + 1));
        btnRecognize.setOnClickListener(v -> showDetectMenu());

        // 长按图片：弹出操作菜单（保存 / 三项检测）
        ivFullImage.setOnLongClickListener(v -> {
            showDetectMenu();
            return true;
        });

        showImage(currentIndex);
    }

    private void showImage(int index) {
        if (index < 0 || index >= imagePaths.size()) return;
        currentIndex = index;

        btnPrev.setEnabled(index > 0);
        btnNext.setEnabled(index < imagePaths.size() - 1);

        tvViewerIndex.setText(getString(R.string.viewer_index, index + 1, imagePaths.size()));

        File file = new File(imagePaths.get(index));
        tvViewerTitle.setText(file.getName());

        ivFullImage.setImageResource(android.R.color.black);

        executor.execute(() -> {
            Bitmap bitmap = decodeBitmap(imagePaths.get(index));
            mainHandler.post(() -> {
                if (bitmap != null && !isFinishing()) {
                    ivFullImage.setImageBitmap(bitmap);
                }
            });
        });
    }

    /** 长按图片 / 点“检测”按钮：操作菜单（保存 + 三项检测） */
    private void showDetectMenu() {
        final String[] items = {"💾 保存到相册", "🔍 二维码识别", "🎯 YOLO 目标检测", "🔧 万用表读数识别"};
        new AlertDialog.Builder(this)
                .setTitle("选择操作")
                .setItems(items, (d, which) -> {
                    if (which == 0) {
                        saveToGallery();
                    } else {
                        runDetect(which - 1);
                    }
                })
                .setNegativeButton("取消", null)
                .show();
    }

    /** 按类型执行检测（后台线程） */
    private void runDetect(int type) {
        if (imagePaths == null || currentIndex < 0 || currentIndex >= imagePaths.size()) return;
        final String path = imagePaths.get(currentIndex);
        btnRecognize.setEnabled(false);
        btnRecognize.setText("检测中...");

        executor.execute(() -> {
            mAnnotatedBitmap = null;
            final String title;
            final String message;
            switch (type) {
                case DETECT_YOLO:
                    title = "YOLO 目标检测";
                    message = doYoloDetect(path);
                    break;
                case DETECT_METER:
                    title = "万用表读数识别";
                    message = doMeterRecognize(path);
                    break;
                case DETECT_QR:
                default:
                    title = "二维码内容";
                    message = doQrRecognize(path);
                    break;
            }
            mainHandler.post(() -> {
                if (isFinishing()) return;
                btnRecognize.setEnabled(true);
                btnRecognize.setText(R.string.viewer_btn_detect);
                if (type == DETECT_YOLO && mAnnotatedBitmap != null) {
                    ivFullImage.setImageBitmap(mAnnotatedBitmap);
                }
                showResultDialog(title, message);
            });
        });
    }

    /** 二维码识别：返回结果文本（null 表示提示已由 Toast 处理，如解码失败以外的情况） */
    private String doQrRecognize(String path) {
        Bitmap bitmap = decodeForRecognition(path);
        String result = null;
        try {
            result = Vision.get().decodeQrCode(bitmap);
        } catch (Exception ignored) {
        }
        if (bitmap != null) bitmap.recycle();
        if (result == null || result.isEmpty()) {
            return "未识别到二维码\n\n请尝试拍摄更清晰、对焦、正对二维码的图片。";
        }
        return result;
    }

    /** YOLO 检测：返回目标清单文本（类别 × 置信度），并在图片上绘制检测框 */
    private String doYoloDetect(String path) {
        YoloDetector detector = YoloDetectorHolder.get(getApplicationContext());
        if (detector == null || !detector.isReady()) {
            String reason = YoloDetectorHolder.getInitError();
            mAnnotatedBitmap = null;
            return "模型加载失败\n\n" + (reason != null && !reason.isEmpty() ? reason : "详见 logcat");
        }
        Bitmap bitmap = decodeForRecognition(path);
        if (bitmap == null) {
            mAnnotatedBitmap = null;
            return "照片解码失败";
        }
        long t0 = System.currentTimeMillis();
        List<YoloDetector.Detection> dets = detector.detect(bitmap);
        long ms = System.currentTimeMillis() - t0;

        // 在原图内存副本上绘制检测框（不写磁盘，磁盘原图保持不变）
        mAnnotatedBitmap = drawDetections(bitmap, dets);

        StringBuilder sb = new StringBuilder();
        sb.append(String.format(Locale.US, "检测到 %d 个目标，耗时 %dms\n\n",
                dets != null ? dets.size() : 0, ms));
        if (dets != null && !dets.isEmpty()) {
            for (YoloDetector.Detection d : dets) {
                sb.append(String.format(Locale.US, "· %s  置信度 %.0f%%\n",
                        d.className, d.score * 100));
            }
        }
        return sb.toString();
    }

    /** 在图片上绘制 YOLO 检测框与标签（原地绘制内存副本，不影响磁盘原图） */
    private Bitmap drawDetections(Bitmap src, List<YoloDetector.Detection> dets) {
        if (src == null) return null;
        Canvas canvas = new Canvas(src);
        float w = src.getWidth();
        float h = src.getHeight();

        float stroke = Math.max(2f, w / 400f);
        float textSize = Math.max(28f, w / 40f);

        Paint boxPaint = new Paint();
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(stroke);
        boxPaint.setColor(Color.parseColor("#FF3B30"));
        boxPaint.setAntiAlias(true);

        Paint textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(textSize);
        textPaint.setAntiAlias(true);

        Paint bgPaint = new Paint();
        bgPaint.setStyle(Paint.Style.FILL);
        bgPaint.setColor(Color.parseColor("#CCFF3B30"));

        if (dets != null) {
            for (YoloDetector.Detection d : dets) {
                float x1 = d.x1 * w;
                float y1 = d.y1 * h;
                float x2 = d.x2 * w;
                float y2 = d.y2 * h;
                canvas.drawRect(x1, y1, x2, y2, boxPaint);

                String label = d.className + " " + Math.round(d.score * 100) + "%";
                float tw = textPaint.measureText(label);
                float th = textPaint.getTextSize();
                float labelTop = y1 - th - stroke;
                if (labelTop < 0) labelTop = y1;
                canvas.drawRect(x1, labelTop, x1 + tw + 8, labelTop + th + 4, bgPaint);
                canvas.drawText(label, x1 + 4, labelTop + th, textPaint);
            }
        }
        return src;
    }

    /** 万用表读数识别：云端识别读数与挡位 */
    private String doMeterRecognize(String path) {
        Bitmap bitmap = decodeForRecognition(path);
        if (bitmap == null) return "照片解码失败";
        MeterReading reading = null;
        String error = null;
        try {
            reading = Vision.get().readMeter(bitmap);
        } catch (Exception e) {
            error = "识别异常：" + e.getMessage();
        }
        bitmap.recycle();
        if (reading != null && reading.hasValue()) {
            return "识别结果：\n\n" + reading.getDisplayText();
        }
        return "识别失败\n\n" + (error != null ? error
                : (MeterCloudOcr.getLastError() != null
                ? MeterCloudOcr.getLastError() : "未识别到有效读数"));
    }

    /** 保存当前图片到系统相册（MediaStore，兼容 Android 10+ 与旧版本） */
    private void saveToGallery() {
        if (imagePaths == null || currentIndex < 0 || currentIndex >= imagePaths.size()) return;
        final String path = imagePaths.get(currentIndex);
        final String name = new File(path).getName();
        btnRecognize.setEnabled(false);
        btnRecognize.setText("保存中...");

        executor.execute(() -> {
            String error = null;
            try {
                ContentValues values = new ContentValues();
                values.put(MediaStore.Images.Media.DISPLAY_NAME, name);
                values.put(MediaStore.Images.Media.MIME_TYPE, guessMime(name));
                Uri uri;
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    values.put(MediaStore.Images.Media.RELATIVE_PATH,
                            Environment.DIRECTORY_PICTURES + "/ARGlass");
                    uri = getContentResolver().insert(
                            MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
                } else {
                    values.put(MediaStore.Images.Media.DATA, path);
                    uri = getContentResolver().insert(
                            MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
                }
                if (uri == null) {
                    error = "无法写入相册";
                } else {
                    try (OutputStream os = getContentResolver().openOutputStream(uri)) {
                        if (os == null) {
                            error = "无法打开相册输出流";
                        } else {
                            byte[] data = readFileBytes(path);
                            if (data == null) {
                                error = "读取原图失败";
                            } else {
                                os.write(data);
                                os.flush();
                            }
                        }
                    }
                }
            } catch (Exception e) {
                error = e.getMessage();
            }
            final String fError = error;
            mainHandler.post(() -> {
                if (isFinishing()) return;
                btnRecognize.setEnabled(true);
                btnRecognize.setText("🔍 检测");
                Toast.makeText(this, fError == null
                        ? "已保存到相册 Pictures/ARGlass" : ("保存失败：" + fError), Toast.LENGTH_SHORT).show();
            });
        });
    }

    private static String guessMime(String name) {
        String n = name.toLowerCase();
        if (n.endsWith(".png")) return "image/png";
        if (n.endsWith(".webp")) return "image/webp";
        if (n.endsWith(".bmp")) return "image/bmp";
        return "image/jpeg";
    }

    private static byte[] readFileBytes(String path) {
        try (java.io.FileInputStream fis = new java.io.FileInputStream(path)) {
            java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream();
            byte[] buf = new byte[8192];
            int len;
            while ((len = fis.read(buf)) > 0) bos.write(buf, 0, len);
            return bos.toByteArray();
        } catch (Exception e) {
            return null;
        }
    }

    private Bitmap decodeForRecognition(String path) {
        try {
            BitmapFactory.Options opts = new BitmapFactory.Options();
            opts.inJustDecodeBounds = true;
            BitmapFactory.decodeFile(path, opts);

            int reqSize = 2500;
            int sample = 1;
            int max = Math.max(opts.outWidth, opts.outHeight);
            while (max / sample > reqSize) sample *= 2;

            opts.inSampleSize = sample;
            opts.inJustDecodeBounds = false;
            opts.inPreferredConfig = Bitmap.Config.ARGB_8888;
            return BitmapFactory.decodeFile(path, opts);
        } catch (Exception e) {
            return null;
        }
    }

    private void showResultDialog(String title, String message) {
        new AlertDialog.Builder(this)
                .setTitle(title)
                .setMessage(message)
                .setPositiveButton("确定", null)
                .show();
    }

    private Bitmap decodeBitmap(String path) {
        try {
            // 获取屏幕尺寸来计算采样率
            int screenW = getResources().getDisplayMetrics().widthPixels;
            int screenH = getResources().getDisplayMetrics().heightPixels;

            BitmapFactory.Options opts = new BitmapFactory.Options();
            opts.inJustDecodeBounds = true;
            BitmapFactory.decodeFile(path, opts);

            opts.inSampleSize = calculateInSampleSize(opts, screenW, screenH);
            opts.inJustDecodeBounds = false;
            opts.inPreferredConfig = Bitmap.Config.RGB_565;
            return BitmapFactory.decodeFile(path, opts);
        } catch (OutOfMemoryError e) {
            // 如果还是OOM，加大采样率
            try {
                BitmapFactory.Options opts = new BitmapFactory.Options();
                opts.inSampleSize = 4;
                opts.inPreferredConfig = Bitmap.Config.RGB_565;
                return BitmapFactory.decodeFile(path, opts);
            } catch (Exception ex) {
                return null;
            }
        } catch (Exception e) {
            return null;
        }
    }

    private static int calculateInSampleSize(BitmapFactory.Options options, int reqWidth, int reqHeight) {
        final int height = options.outHeight;
        final int width = options.outWidth;
        int inSampleSize = 1;
        if (height > reqHeight || width > reqWidth) {
            final int halfHeight = height / 2;
            final int halfWidth = width / 2;
            while ((halfHeight / inSampleSize) >= reqHeight
                    && (halfWidth / inSampleSize) >= reqWidth) {
                inSampleSize *= 2;
            }
        }
        return inSampleSize;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.shutdownNow();
    }
}
