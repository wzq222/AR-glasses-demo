package com.ar.glass.ui;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import com.ar.glass.R;
import com.ar.glass.vision.Vision;

import java.io.File;
import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * 全屏图片查看器
 */
public class ImageViewerActivity extends AppCompatActivity {

    private ImageView ivFullImage;
    private TextView tvViewerTitle;
    private TextView tvViewerIndex;
    private Button btnViewerBack;
    private Button btnPrev;
    private Button btnNext;
    private Button btnRecognize;

    private ArrayList<String> imagePaths;
    private int currentIndex = 0;
    private ExecutorService executor = Executors.newSingleThreadExecutor();
    private Handler mainHandler = new Handler(Looper.getMainLooper());

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
        btnRecognize.setOnClickListener(v -> recognizeQrCode());

        showImage(currentIndex);
    }

    private void showImage(int index) {
        if (index < 0 || index >= imagePaths.size()) return;
        currentIndex = index;

        btnPrev.setEnabled(index > 0);
        btnNext.setEnabled(index < imagePaths.size() - 1);

        tvViewerIndex.setText((index + 1) + "/" + imagePaths.size());

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

    private void recognizeQrCode() {
        if (imagePaths == null || currentIndex < 0 || currentIndex >= imagePaths.size()) return;
        final String path = imagePaths.get(currentIndex);
        btnRecognize.setEnabled(false);
        btnRecognize.setText("识别中...");

        executor.execute(() -> {
            Bitmap bitmap = decodeForRecognition(path);
            final String result = Vision.get().decodeQrCode(bitmap);
            if (bitmap != null) bitmap.recycle();

            mainHandler.post(() -> {
                if (isFinishing()) return;
                btnRecognize.setEnabled(true);
                btnRecognize.setText("识别二维码");
                if (result == null || result.isEmpty()) {
                    showResultDialog("未识别到二维码", "请尝试拍摄更清晰、对焦、正对二维码的图片。");
                } else {
                    showResultDialog("二维码内容", result);
                }
            });
        });
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
