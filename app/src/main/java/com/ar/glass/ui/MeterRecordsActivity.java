package com.ar.glass.ui;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.util.LruCache;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import com.ar.glass.R;
import com.ar.glass.record.MeterRecord;
import com.ar.glass.record.MeterRecordStore;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * 巡检台账：查看历史读数记录、导出 CSV/Excel、查看/删除单条记录。
 */
public class MeterRecordsActivity extends AppCompatActivity {

    private static final String TAG = "MeterRecords";
    private static final String FILE_PROVIDER_AUTHORITY = "com.ar.glass.fileprovider";

    private TextView tvCount;
    private TextView tvEmpty;
    private ListView lvRecords;
    private Button btnExportCsv;
    private Button btnExportExcel;
    private Button btnClear;
    private Button btnBack;

    private MeterRecordStore store;
    private RecordAdapter adapter;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private LruCache<String, Bitmap> thumbCache;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_meter_records);

        store = MeterRecordStore.get(this);

        final int maxMemory = (int) (Runtime.getRuntime().maxMemory() / 1024);
        thumbCache = new LruCache<String, Bitmap>(maxMemory / 8) {
            @Override
            protected int sizeOf(String key, Bitmap bitmap) {
                return bitmap.getByteCount() / 1024;
            }
        };

        tvCount = findViewById(R.id.tvCount);
        tvEmpty = findViewById(R.id.tvEmpty);
        lvRecords = findViewById(R.id.lvRecords);
        btnExportCsv = findViewById(R.id.btnExportCsv);
        btnExportExcel = findViewById(R.id.btnExportExcel);
        btnClear = findViewById(R.id.btnClear);
        btnBack = findViewById(R.id.btnBack);

        btnBack.setOnClickListener(v -> finish());
        btnExportCsv.setOnClickListener(v -> doExport(false));
        btnExportExcel.setOnClickListener(v -> doExport(true));
        btnClear.setOnClickListener(v -> confirmClear());

        lvRecords.setOnItemClickListener((parent, view, position, id) -> {
            if (adapter != null) {
                showRecordDetail((MeterRecord) adapter.getItem(position));
            }
        });

        reload();
    }

    private void reload() {
        List<MeterRecord> list = store.getAll();
        tvCount.setText("共 " + list.size() + " 条记录");
        if (list.isEmpty()) {
            tvEmpty.setVisibility(View.VISIBLE);
            lvRecords.setVisibility(View.GONE);
        } else {
            tvEmpty.setVisibility(View.GONE);
            lvRecords.setVisibility(View.VISIBLE);
            adapter = new RecordAdapter(list);
            lvRecords.setAdapter(adapter);
        }
    }

    private void doExport(final boolean excel) {
        Toast.makeText(this, "正在导出...", Toast.LENGTH_SHORT).show();
        executor.execute(() -> {
            final File f = excel ? store.exportExcel() : store.exportCsv();
            runOnUiThread(() -> {
                if (f == null || !f.exists()) {
                    Toast.makeText(this, "导出失败", Toast.LENGTH_SHORT).show();
                    return;
                }
                Toast.makeText(this, "已导出：" + f.getName(), Toast.LENGTH_LONG).show();
                shareFile(f);
            });
        });
    }

    /** 通过系统分享导出文件；分享失败则提示文件保存路径。 */
    private void shareFile(File f) {
        try {
            Uri uri = FileProvider.getUriForFile(this, FILE_PROVIDER_AUTHORITY, f);
            Intent send = new Intent(Intent.ACTION_SEND);
            send.setType(f.getName().endsWith(".csv") ? "text/csv" : "application/vnd.ms-excel");
            send.putExtra(Intent.EXTRA_STREAM, uri);
            send.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            startActivity(Intent.createChooser(send, "导出巡检记录"));
        } catch (Exception e) {
            Toast.makeText(this, "文件已保存到：\n" + f.getAbsolutePath(), Toast.LENGTH_LONG).show();
        }
    }

    private void confirmClear() {
        if (store.size() == 0) {
            Toast.makeText(this, "台账已为空", Toast.LENGTH_SHORT).show();
            return;
        }
        new AlertDialog.Builder(this)
                .setTitle("清空台账")
                .setMessage("确定要删除全部巡检记录吗？此操作不可恢复。")
                .setPositiveButton("清空", (d, w) -> {
                    store.clear();
                    reload();
                    Toast.makeText(this, "已清空", Toast.LENGTH_SHORT).show();
                })
                .setNegativeButton("取消", null)
                .show();
    }

    /** 详情弹窗：完整信息 + 原照片 + 删除。 */
    private void showRecordDetail(final MeterRecord r) {
        LinearLayout container = new LinearLayout(this);
        container.setOrientation(LinearLayout.VERTICAL);
        int pad = dp(24);
        container.setPadding(pad, dp(12), pad, dp(4));

        TextView tvReading = new TextView(this);
        tvReading.setText(r.getReadingText());
        tvReading.setTextSize(26);
        tvReading.setTextColor(Color.rgb(51, 51, 51));
        tvReading.setGravity(Gravity.CENTER);
        container.addView(tvReading);

        TextView tvMeta = new TextView(this);
        StringBuilder meta = new StringBuilder();
        meta.append("时间：").append(r.getTimeText());
        if (r.gear != null && !r.gear.isEmpty()) {
            meta.append("\n挡位：").append(r.gear);
        }
        if (r.unit != null && !r.unit.isEmpty()) {
            meta.append("\n单位：").append(r.unit);
        }
        if (r.warning != null && !r.warning.isEmpty()) {
            meta.append("\n⚠️ ").append(r.warning);
        }
        tvMeta.setText(meta.toString());
        tvMeta.setTextSize(14);
        tvMeta.setTextColor(Color.rgb(102, 102, 102));
        tvMeta.setPadding(0, dp(14), 0, dp(6));
        container.addView(tvMeta);

        if (r.photoPath != null && !r.photoPath.isEmpty() && new File(r.photoPath).exists()) {
            ImageView iv = new ImageView(this);
            Bitmap b = decodeSampledBitmap(r.photoPath, 900, 900);
            if (b != null) {
                iv.setImageBitmap(b);
                iv.setScaleType(ImageView.ScaleType.FIT_CENTER);
                LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT, dp(360));
                iv.setLayoutParams(lp);
                container.addView(iv);
            }
        }

        new AlertDialog.Builder(this)
                .setTitle("巡检记录详情")
                .setView(container)
                .setPositiveButton("关闭", null)
                .setNegativeButton("删除", (d, w) -> {
                    store.delete(r);
                    reload();
                })
                .show();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.shutdownNow();
        if (thumbCache != null) {
            thumbCache.evictAll();
        }
    }

    private int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density + 0.5f);
    }

    /**
     * 记录列表适配器：读数 + 挡位 + 时间 + 照片缩略图。
     */
    private class RecordAdapter extends BaseAdapter {
        private final List<MeterRecord> list;
        private final LayoutInflater inflater;

        RecordAdapter(List<MeterRecord> list) {
            this.list = new ArrayList<>(list);
            this.inflater = LayoutInflater.from(MeterRecordsActivity.this);
        }

        @Override
        public int getCount() {
            return list.size();
        }

        @Override
        public Object getItem(int position) {
            return list.get(position);
        }

        @Override
        public long getItemId(int position) {
            return position;
        }

        @Override
        public View getView(int position, View convertView, ViewGroup parent) {
            ViewHolder holder;
            if (convertView == null) {
                convertView = inflater.inflate(R.layout.item_meter_record, parent, false);
                holder = new ViewHolder();
                holder.tvReading = convertView.findViewById(R.id.tvReading);
                holder.tvMeta = convertView.findViewById(R.id.tvMeta);
                holder.tvTime = convertView.findViewById(R.id.tvTime);
                holder.ivThumb = convertView.findViewById(R.id.ivThumb);
                convertView.setTag(holder);
            } else {
                holder = (ViewHolder) convertView.getTag();
            }

            MeterRecord r = list.get(position);
            holder.tvReading.setText(r.getReadingText());

            StringBuilder meta = new StringBuilder();
            if (r.gear != null && !r.gear.isEmpty()) {
                meta.append("挡位：").append(r.gear);
            }
            if (r.warning != null && !r.warning.isEmpty()) {
                if (meta.length() > 0) {
                    meta.append("　");
                }
                meta.append("⚠️ ").append(r.warning);
            }
            holder.tvMeta.setText(meta.length() > 0 ? meta.toString() : "—");
            holder.tvTime.setText(r.getTimeText());

            // 照片缩略图（同步解码 + 缓存，记录量小、缩略图小，开销可忽略）
            holder.ivThumb.setVisibility(View.GONE);
            if (r.photoPath != null && !r.photoPath.isEmpty()) {
                File pf = new File(r.photoPath);
                if (pf.exists()) {
                    holder.ivThumb.setVisibility(View.VISIBLE);
                    Bitmap cached = thumbCache.get(r.photoPath);
                    if (cached != null && !cached.isRecycled()) {
                        holder.ivThumb.setImageBitmap(cached);
                    } else {
                        Bitmap thumb = decodeSampledBitmap(r.photoPath, 112, 112);
                        if (thumb != null) {
                            thumbCache.put(r.photoPath, thumb);
                            holder.ivThumb.setImageBitmap(thumb);
                        }
                    }
                }
            }

            return convertView;
        }

        private class ViewHolder {
            TextView tvReading;
            TextView tvMeta;
            TextView tvTime;
            ImageView ivThumb;
        }
    }

    /** 按采样率解码图片，避免 OOM。 */
    private static Bitmap decodeSampledBitmap(String path, int reqWidth, int reqHeight) {
        try {
            final BitmapFactory.Options options = new BitmapFactory.Options();
            options.inJustDecodeBounds = true;
            BitmapFactory.decodeFile(path, options);
            options.inSampleSize = calculateInSampleSize(options, reqWidth, reqHeight);
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
