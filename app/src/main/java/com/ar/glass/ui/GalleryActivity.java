package com.ar.glass.ui;

import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.OpenableColumns;
import android.os.Handler;
import android.os.Looper;
import android.util.DisplayMetrics;
import android.util.LruCache;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.Button;
import android.widget.GridView;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import com.ar.glass.R;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * 图片浏览Activity
 * 浏览原图（glass_media/photos目录）
 */
public class GalleryActivity extends AppCompatActivity {

    public static final String EXTRA_MODE = "gallery_mode";
    public static final String MODE_ORIGINAL = "original";
    public static final String EXTRA_SELECT_IMAGE = "select_image";
    public static final String EXTRA_SELECTED_IMAGE_PATH = "selected_image_path";

    private GridView gridImages;
    private TextView tvGalleryTitle;
    private TextView tvImageCount;
    private TextView tvEmpty;
    private Button btnBack;
    private Button btnImport;

    private final ActivityResultLauncher<String> importLauncher =
            registerForActivityResult(new ActivityResultContracts.GetMultipleContents(), this::importImages);

    private File rootDir;
    private List<File> imageFiles = new ArrayList<>();
    private ImageAdapter adapter;
    private LruCache<String, Bitmap> thumbnailCache;
    private ExecutorService executor = Executors.newFixedThreadPool(4);
    private Handler mainHandler = new Handler(Looper.getMainLooper());
    private boolean selectImageMode;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_gallery);
        selectImageMode = getIntent().getBooleanExtra(EXTRA_SELECT_IMAGE, false);

        // 初始化缩略图缓存
        final int maxMemory = (int) (Runtime.getRuntime().maxMemory() / 1024);
        final int cacheSize = maxMemory / 8;
        thumbnailCache = new LruCache<String, Bitmap>(cacheSize) {
            @Override
            protected int sizeOf(String key, Bitmap bitmap) {
                return bitmap.getByteCount() / 1024;
            }
        };

        initViews();

        setupDirectory();
        loadImages();
    }

    private void initViews() {
        gridImages = findViewById(R.id.gridImages);
        tvGalleryTitle = findViewById(R.id.tvGalleryTitle);
        tvImageCount = findViewById(R.id.tvImageCount);
        tvEmpty = findViewById(R.id.tvEmpty);
        btnBack = findViewById(R.id.btnBack);
        btnImport = findViewById(R.id.btnImport);

        btnBack.setOnClickListener(v -> finish());
        btnImport.setOnClickListener(v -> importLauncher.launch("image/*"));

        gridImages.setOnItemClickListener((parent, view, position, id) -> {
            if (position >= 0 && position < imageFiles.size()) {
                openOrReturnImage(position);
            }
        });
    }

    private void setupDirectory() {
        File externalDir = getExternalFilesDir(null);
        if (externalDir == null) {
            Toast.makeText(this, "无法访问外部存储", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        rootDir = new File(externalDir, "glass_media/photos");
        tvGalleryTitle.setText(selectImageMode ? "选择原图库图片" : "原图库 (眼镜原始照片)");
        if (selectImageMode) tvEmpty.setText("原图库为空，可点右上角导入图片");
    }

    private void loadImages() {
        if (rootDir == null || !rootDir.exists()) {
            tvEmpty.setVisibility(View.VISIBLE);
            tvImageCount.setText("0张");
            return;
        }

        executor.execute(() -> {
            List<File> files = new ArrayList<>();
            collectImages(rootDir, files);

            // 按修改时间排序（最新的在前）
            Collections.sort(files, (f1, f2) -> Long.compare(f2.lastModified(), f1.lastModified()));

            mainHandler.post(() -> {
                imageFiles = files;
                if (files.isEmpty()) {
                    tvEmpty.setVisibility(View.VISIBLE);
                    tvImageCount.setText("0张");
                } else {
                    tvEmpty.setVisibility(View.GONE);
                    tvImageCount.setText(files.size() + "张");
                    adapter = new ImageAdapter(this, files);
                    gridImages.setAdapter(adapter);
                }
            });
        });
    }

    /**
     * 递归收集目录下所有图片文件
     */
    private void collectImages(File dir, List<File> files) {
        if (dir == null || !dir.exists()) return;
        File[] list = dir.listFiles();
        if (list == null) return;

        for (File f : list) {
            if (f.isDirectory()) {
                collectImages(f, files);
            } else {
                String name = f.getName().toLowerCase();
                boolean supportedEvidence = name.endsWith(".jpg") || name.endsWith(".jpeg")
                        || name.endsWith(".png") || name.endsWith(".webp");
                if (supportedEvidence || (!selectImageMode && name.endsWith(".bmp"))) {
                    files.add(f);
                }
            }
        }
    }

    private void openImageViewer(int index) {
        Intent intent = new Intent(this, ImageViewerActivity.class);
        // 传递所有图片路径
        ArrayList<String> paths = new ArrayList<>();
        for (File f : imageFiles) {
            paths.add(f.getAbsolutePath());
        }
        intent.putStringArrayListExtra("image_paths", paths);
        intent.putExtra("start_index", index);
        intent.putExtra("title", tvGalleryTitle.getText().toString());
        startActivity(intent);
    }

    private void openOrReturnImage(int position) {
        if (selectImageMode) {
            returnSelectedImage(position);
        } else {
            openImageViewer(position);
        }
    }

    private void returnSelectedImage(int position) {
        Intent result = new Intent().putExtra(
                EXTRA_SELECTED_IMAGE_PATH, imageFiles.get(position).getAbsolutePath());
        setResult(RESULT_OK, result);
        finish();
    }

    /** 从手机相册导入图片到原图库 */
    private void importImages(List<Uri> uris) {
        if (uris == null || uris.isEmpty()) return;
        if (rootDir == null) return;
        if (!rootDir.exists() && !rootDir.mkdirs()) {
            Toast.makeText(this, "无法创建照片目录", Toast.LENGTH_SHORT).show();
            return;
        }
        executor.execute(() -> {
            int imported = 0;
            for (Uri uri : uris) {
                if (copyImageToPhotos(uri)) imported++;
            }
            final int count = imported;
            mainHandler.post(() -> {
                if (count > 0) {
                    Toast.makeText(this, "已导入 " + count + " 张图片", Toast.LENGTH_SHORT).show();
                    loadImages();
                } else {
                    Toast.makeText(this, "导入失败，请重试", Toast.LENGTH_SHORT).show();
                }
            });
        });
    }

    private boolean copyImageToPhotos(Uri uri) {
        try {
            ContentResolver resolver = getContentResolver();
            String name = queryDisplayName(resolver, uri);
            if (name == null || name.isEmpty()) {
                name = "import_" + System.currentTimeMillis() + ".jpg";
            }
            File out = uniqueFile(rootDir, name);
            InputStream in = resolver.openInputStream(uri);
            if (in == null) return false;
            FileOutputStream outStream = new FileOutputStream(out);
            byte[] buf = new byte[8192];
            int len;
            while ((len = in.read(buf)) > 0) {
                outStream.write(buf, 0, len);
            }
            outStream.flush();
            outStream.close();
            in.close();
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private String queryDisplayName(ContentResolver resolver, Uri uri) {
        Cursor cursor = null;
        try {
            cursor = resolver.query(uri, new String[]{OpenableColumns.DISPLAY_NAME}, null, null, null);
            if (cursor != null && cursor.moveToFirst()) {
                int idx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (idx >= 0) return cursor.getString(idx);
            }
            return null;
        } catch (Exception e) {
            return null;
        } finally {
            if (cursor != null) cursor.close();
        }
    }

    /** 生成不重名的目标文件 */
    private File uniqueFile(File dir, String name) {
        File f = new File(dir, name);
        if (!f.exists()) return f;
        int dot = name.lastIndexOf('.');
        String base = dot > 0 ? name.substring(0, dot) : name;
        String ext = dot > 0 ? name.substring(dot) : "";
        int i = 1;
        while (f.exists()) {
            f = new File(dir, base + "_" + i + ext);
            i++;
        }
        return f;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.shutdownNow();
        thumbnailCache.evictAll();
    }

    /**
     * 图片网格适配器
     */
    private class ImageAdapter extends BaseAdapter {
        private Context context;
        private List<File> files;
        private LayoutInflater inflater;
        private int thumbSize;

        ImageAdapter(Context context, List<File> files) {
            this.context = context;
            this.files = files;
            this.inflater = LayoutInflater.from(context);

            // 计算缩略图尺寸（屏幕宽度/3，考虑间距）
            DisplayMetrics dm = context.getResources().getDisplayMetrics();
            thumbSize = (dm.widthPixels - 32) / 3; // 3列，padding+spacing约32px
        }

        @Override
        public int getCount() {
            return files.size();
        }

        @Override
        public Object getItem(int position) {
            return files.get(position);
        }

        @Override
        public long getItemId(int position) {
            return position;
        }

        @Override
        public View getView(int position, View convertView, ViewGroup parent) {
            ImageView imageView;
            if (convertView == null) {
                View itemView = inflater.inflate(R.layout.item_gallery_thumb, parent, false);
                imageView = itemView.findViewById(R.id.ivThumb);
                // 设置固定尺寸
                ViewGroup.LayoutParams params = imageView.getLayoutParams();
                params.height = thumbSize;
                imageView.setLayoutParams(params);
                itemView.setTag(imageView);
                convertView = itemView;
            } else {
                imageView = (ImageView) convertView.getTag();
            }

            final File imageFile = files.get(position);
            final String path = imageFile.getAbsolutePath();

            // 记录当前 imageView 对应的图片路径，供异步回调判断是否已被复用
            imageView.setTag(path);

            // 先显示占位色
            imageView.setImageResource(android.R.color.darker_gray);

            // 尝试从缓存获取
            Bitmap cached = thumbnailCache.get(path);
            if (cached != null && !cached.isRecycled()) {
                imageView.setImageBitmap(cached);
            } else {
                // 异步加载缩略图
                executor.execute(() -> {
                    Bitmap thumb = decodeSampledBitmap(path, thumbSize, thumbSize);
                    if (thumb != null) {
                        thumbnailCache.put(path, thumb);
                        mainHandler.post(() -> {
                            // 检查 imageView 是否仍显示同一张图片（未被复用）
                            if (path.equals(imageView.getTag())) {
                                imageView.setImageBitmap(thumb);
                            }
                        });
                    }
                });
            }

            return convertView;
        }
    }

    /**
     * 按采样率解码图片生成缩略图，避免OOM
     */
    private static Bitmap decodeSampledBitmap(String path, int reqWidth, int reqHeight) {
        try {
            // 先获取图片尺寸
            final BitmapFactory.Options options = new BitmapFactory.Options();
            options.inJustDecodeBounds = true;
            BitmapFactory.decodeFile(path, options);

            // 计算采样率
            options.inSampleSize = calculateInSampleSize(options, reqWidth, reqHeight);

            // 解码缩略图
            options.inJustDecodeBounds = false;
            options.inPreferredConfig = Bitmap.Config.RGB_565;
            return BitmapFactory.decodeFile(path, options);
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
}
