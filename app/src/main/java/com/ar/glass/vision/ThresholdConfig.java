package com.ar.glass.vision;

import android.content.Context;
import android.content.SharedPreferences;

/**
 * 阈值配置：按量纲类别（电压/电流/电阻/电容/频率）持久化上下限。
 *
 * 存储：SharedPreferences（应用私有，无需额外权限）。
 * 每个类别可独立启用/关闭上限与下限；默认全部关闭（不报警）。
 */
public class ThresholdConfig {

    public static final String CATEGORY_VOLTAGE = "电压";
    public static final String CATEGORY_CURRENT = "电流";
    public static final String CATEGORY_RESISTANCE = "电阻";
    public static final String CATEGORY_CAPACITANCE = "电容";
    public static final String CATEGORY_FREQUENCY = "频率";

    /** 支持配置的类别顺序（用于设置 UI 展示）。 */
    public static final String[] CATEGORIES = {
            CATEGORY_VOLTAGE, CATEGORY_CURRENT, CATEGORY_RESISTANCE,
            CATEGORY_CAPACITANCE, CATEGORY_FREQUENCY
    };

    private static final String PREFS = "meter_thresholds";

    private final SharedPreferences sp;

    /** 某类别的上下限配置。 */
    public static class Bounds {
        public boolean upperEnabled;
        public double upper;
        public boolean lowerEnabled;
        public double lower;
    }

    public ThresholdConfig(Context context) {
        sp = context.getApplicationContext().getSharedPreferences(PREFS, Context.MODE_PRIVATE);
    }

    /** 读取某类别的上下限（未配置任何一项也返回对象，但 enabled 均为 false）。 */
    public Bounds getBounds(String category) {
        Bounds b = new Bounds();
        b.upperEnabled = sp.getBoolean(keyUpperEnabled(category), false);
        b.upper = Double.longBitsToDouble(sp.getLong(keyUpper(category), Double.doubleToLongBits(Double.NaN)));
        b.lowerEnabled = sp.getBoolean(keyLowerEnabled(category), false);
        b.lower = Double.longBitsToDouble(sp.getLong(keyLower(category), Double.doubleToLongBits(Double.NaN)));
        return b;
    }

    /** 保存某类别的上下限。 */
    public void setBounds(String category, boolean upperEnabled, double upper,
                          boolean lowerEnabled, double lower) {
        SharedPreferences.Editor e = sp.edit();
        e.putBoolean(keyUpperEnabled(category), upperEnabled);
        e.putLong(keyUpper(category), Double.doubleToLongBits(upper));
        e.putBoolean(keyLowerEnabled(category), lowerEnabled);
        e.putLong(keyLower(category), Double.doubleToLongBits(lower));
        e.apply();
    }

    /** 是否开启了任意一项阈值（用于主界面开关状态提示）。 */
    public boolean anyEnabled() {
        for (String c : CATEGORIES) {
            Bounds b = getBounds(c);
            if (b.upperEnabled || b.lowerEnabled) {
                return true;
            }
        }
        return false;
    }

    private static String keyUpperEnabled(String c) { return c + "_upper_enabled"; }
    private static String keyUpper(String c) { return c + "_upper"; }
    private static String keyLowerEnabled(String c) { return c + "_lower_enabled"; }
    private static String keyLower(String c) { return c + "_lower"; }
}
