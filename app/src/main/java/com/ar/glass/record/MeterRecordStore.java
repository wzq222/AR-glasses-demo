package com.ar.glass.record;

import android.content.Context;
import android.util.Log;

import com.ar.glass.vision.MeterReading;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Locale;

/**
 * 巡检记录台账：JSON 文件持久化 + 原照片归档 + CSV/Excel 导出。
 *
 * 存储位置：
 * - 记录 JSON：app 内部 filesDir/meter_records.json
 * - 原照片：app 外部专属目录 meter_photos/
 * - 导出文件：app 外部专属目录 meter_exports/
 * （均在应用专属目录内，无需额外存储权限）
 */
public class MeterRecordStore {

    private static final String TAG = "MeterRecordStore";
    private static final String RECORDS_FILE = "meter_records.json";
    private static final String PHOTO_DIR = "meter_photos";
    private static final String EXPORT_DIR = "meter_exports";

    private static volatile MeterRecordStore sInstance;

    private final File recordsFile;
    private final File photoDir;
    private final File exportDir;
    private final List<MeterRecord> records = new ArrayList<>();
    private boolean loaded = false;

    private MeterRecordStore(Context context) {
        Context app = context.getApplicationContext();
        recordsFile = new File(app.getFilesDir(), RECORDS_FILE);
        File ext = app.getExternalFilesDir(null);
        if (ext != null) {
            photoDir = new File(ext, PHOTO_DIR);
            exportDir = new File(ext, EXPORT_DIR);
        } else {
            photoDir = new File(app.getFilesDir(), PHOTO_DIR);
            exportDir = new File(app.getFilesDir(), EXPORT_DIR);
        }
    }

    public static MeterRecordStore get(Context context) {
        if (sInstance == null) {
            synchronized (MeterRecordStore.class) {
                if (sInstance == null) {
                    sInstance = new MeterRecordStore(context.getApplicationContext());
                }
            }
        }
        return sInstance;
    }

    /** 导出目录（供外部展示路径用）。 */
    public File getExportDir() {
        if (!exportDir.exists()) {
            exportDir.mkdirs();
        }
        return exportDir;
    }

    /**
     * 新增一条记录并归档原照片。
     *
     * @param reading    识别结果
     * @param photoBytes 原照片字节（可 null，表示不存档照片）
     * @return 新建的记录
     */
    public synchronized MeterRecord add(MeterReading reading, byte[] photoBytes) {
        String photoPath = null;
        if (photoBytes != null && photoBytes.length > 0) {
            photoPath = savePhotoBytes(photoBytes);
        }
        MeterRecord r = new MeterRecord(
                System.currentTimeMillis(),
                reading.value,
                reading.unit,
                reading.gear,
                reading.warning,
                photoPath);
        ensureLoaded();
        records.add(r);
        persist();
        return r;
    }

    /** 获取全部记录（最新在前）。 */
    public synchronized List<MeterRecord> getAll() {
        ensureLoaded();
        List<MeterRecord> copy = new ArrayList<>(records);
        Collections.reverse(copy);
        return copy;
    }

    public synchronized int size() {
        ensureLoaded();
        return records.size();
    }

    /** 删除单条记录（连同其照片文件）。 */
    public synchronized boolean delete(MeterRecord r) {
        ensureLoaded();
        boolean removed = records.remove(r);
        if (removed) {
            persist();
            if (r.photoPath != null && !r.photoPath.isEmpty()) {
                //noinspection ResultOfMethodCallIgnored
                new File(r.photoPath).delete();
            }
        }
        return removed;
    }

    /** 清空全部记录与照片。 */
    public synchronized void clear() {
        records.clear();
        persist();
        File[] files = photoDir.listFiles();
        if (files != null) {
            for (File f : files) {
                //noinspection ResultOfMethodCallIgnored
                f.delete();
            }
        }
    }

    /** 导出 CSV（UTF-8 BOM，Excel 可直接打开），返回生成的文件。 */
    public synchronized File exportCsv() {
        ensureLoaded();
        StringBuilder sb = new StringBuilder();
        sb.append('\ufeff'); // BOM，避免 Excel 打开中文乱码
        sb.append("时间,读数,单位,挡位,异常提示,原照片\n");
        List<MeterRecord> list = getAll();
        for (MeterRecord r : list) {
            sb.append(csv(r.getTimeText())).append(',')
                    .append(csv(r.value)).append(',')
                    .append(csv(r.unit)).append(',')
                    .append(csv(r.gear)).append(',')
                    .append(csv(r.warning)).append(',')
                    .append(csv(r.photoPath)).append('\n');
        }
        return writeExport("巡检记录_" + timestamp() + ".csv", sb.toString());
    }

    /** 导出 Excel（SpreadsheetML 2003 XML 格式，Excel/WPS 直接打开）。 */
    public synchronized File exportExcel() {
        ensureLoaded();
        StringBuilder sb = new StringBuilder();
        sb.append("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        sb.append("<?mso-application progid=\"Excel.Sheet\"?>\n");
        sb.append("<Workbook xmlns=\"urn:schemas-microsoft-com:office:spreadsheet\" ")
                .append("xmlns:ss=\"urn:schemas-microsoft-com:office:spreadsheet\">\n");
        sb.append("<Worksheet ss:Name=\"巡检记录\"><Table>\n");
        sb.append(row(new String[]{"时间", "读数", "单位", "挡位", "异常提示", "原照片"}));
        List<MeterRecord> list = getAll();
        for (MeterRecord r : list) {
            sb.append(row(new String[]{
                    r.getTimeText(), r.value, r.unit, r.gear, r.warning, r.photoPath}));
        }
        sb.append("</Table></Worksheet></Workbook>");
        return writeExport("巡检记录_" + timestamp() + ".xls", sb.toString());
    }

    private void ensureLoaded() {
        if (loaded) {
            return;
        }
        loaded = true;
        records.clear();
        if (!recordsFile.exists()) {
            return;
        }
        try {
            String text = readFile(recordsFile);
            if (text == null || text.trim().isEmpty()) {
                return;
            }
            JSONArray arr = new JSONArray(text);
            for (int i = 0; i < arr.length(); i++) {
                records.add(MeterRecord.fromJson(arr.getJSONObject(i)));
            }
        } catch (Exception e) {
            Log.e(TAG, "加载台账失败", e);
        }
    }

    private void persist() {
        try {
            JSONArray arr = new JSONArray();
            for (MeterRecord r : records) {
                arr.put(r.toJson());
            }
            writeFile(recordsFile, arr.toString());
        } catch (Exception e) {
            Log.e(TAG, "保存台账失败", e);
        }
    }

    private String savePhotoBytes(byte[] bytes) {
        try {
            if (!photoDir.exists()) {
                //noinspection ResultOfMethodCallIgnored
                photoDir.mkdirs();
            }
            File f = new File(photoDir, System.currentTimeMillis() + ".jpg");
            FileOutputStream fos = new FileOutputStream(f);
            fos.write(bytes);
            fos.flush();
            fos.close();
            return f.getAbsolutePath();
        } catch (Exception e) {
            Log.e(TAG, "保存照片失败", e);
            return null;
        }
    }

    private File writeExport(String name, String content) {
        try {
            if (!exportDir.exists()) {
                //noinspection ResultOfMethodCallIgnored
                exportDir.mkdirs();
            }
            File f = new File(exportDir, name);
            writeFile(f, content);
            return f;
        } catch (Exception e) {
            Log.e(TAG, "导出失败", e);
            return null;
        }
    }

    private static String row(String[] cells) {
        StringBuilder sb = new StringBuilder("<Row>");
        for (String c : cells) {
            sb.append("<Cell><Data ss:Type=\"String\">")
                    .append(escapeXml(c))
                    .append("</Data></Cell>");
        }
        sb.append("</Row>\n");
        return sb.toString();
    }

    private static String escapeXml(String s) {
        if (s == null) {
            return "";
        }
        return s.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\"", "&quot;")
                .replace("'", "&apos;");
    }

    private static String csv(String s) {
        if (s == null) {
            return "";
        }
        String v = s;
        if (v.contains(",") || v.contains("\"") || v.contains("\n")) {
            v = "\"" + v.replace("\"", "\"\"") + "\"";
        }
        return v;
    }

    private static String timestamp() {
        return new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
    }

    private static String readFile(File f) throws Exception {
        FileInputStream fis = new FileInputStream(f);
        byte[] b = new byte[(int) f.length()];
        fis.read(b);
        fis.close();
        return new String(b, StandardCharsets.UTF_8);
    }

    private static void writeFile(File f, String s) throws Exception {
        FileOutputStream fos = new FileOutputStream(f);
        OutputStreamWriter w = new OutputStreamWriter(fos, StandardCharsets.UTF_8);
        w.write(s);
        w.flush();
        w.close();
    }
}
