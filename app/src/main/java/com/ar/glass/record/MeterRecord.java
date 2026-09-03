package com.ar.glass.record;

import org.json.JSONObject;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * 一条万用表巡检记录（台账数据模型）。
 *
 * 字段：时间戳、读数、单位、挡位、异常提示、原照片路径。
 * 可序列化为 JSON 持久化，也可用于 CSV/Excel 导出。
 */
public class MeterRecord {

    /** 记录时间（毫秒时间戳）。 */
    public long timestamp;

    /** 数字读数，如 "24.8"。 */
    public String value;

    /** 单位符号，如 "V"、"mA"、"Ω"。 */
    public String unit;

    /** 旋钮挡位，如 "直流电压"、"电阻"。 */
    public String gear;

    /** 异常提示，如 "表笔插孔与挡位不符"。 */
    public String warning;

    /** 原照片本地路径（可空）。 */
    public String photoPath;

    public MeterRecord() {
    }

    public MeterRecord(long timestamp, String value, String unit,
                       String gear, String warning, String photoPath) {
        this.timestamp = timestamp;
        this.value = value;
        this.unit = unit;
        this.gear = gear;
        this.warning = warning;
        this.photoPath = photoPath;
    }

    /** 格式化时间，如 "2026-08-30 17:18:33"。 */
    public String getTimeText() {
        return new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
                .format(new Date(timestamp));
    }

    /** 读数 + 单位，如 "24.8 V"。 */
    public String getReadingText() {
        String v = value == null ? "" : value.trim();
        String u = unit == null ? "" : unit.trim();
        return u.isEmpty() ? v : v + " " + u;
    }

    public JSONObject toJson() {
        JSONObject o = new JSONObject();
        try {
            o.put("ts", timestamp);
            o.put("value", value == null ? "" : value);
            o.put("unit", unit == null ? "" : unit);
            o.put("gear", gear == null ? "" : gear);
            o.put("warning", warning == null ? "" : warning);
            o.put("photo", photoPath == null ? "" : photoPath);
        } catch (Exception ignored) {
        }
        return o;
    }

    public static MeterRecord fromJson(JSONObject o) {
        MeterRecord r = new MeterRecord();
        r.timestamp = o.optLong("ts", System.currentTimeMillis());
        r.value = o.optString("value", "");
        r.unit = o.optString("unit", "");
        r.gear = o.optString("gear", "");
        r.warning = o.optString("warning", "");
        r.photoPath = o.optString("photo", "");
        return r;
    }
}
