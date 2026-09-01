package com.ar.glass.vision;

/**
 * 万用表阈值报警：按量纲类别（电压/电流/电阻等）分别配置上下限，
 * 读数超出范围时给出报警结果。
 *
 * 阈值本身通过 {@link com.ar.glass.vision.ThresholdConfig} 持久化；
 * 本类只负责「把读数换算成标准量 + 判断是否超限」。
 */
public final class ThresholdAlarm {

    /** 超限结果：无报警时状态为 NONE。 */
    public static class Result {
        /** 报警类别（电压/电流/电阻等），无报警为 ""。 */
        public String category = "";
        /** 是否超上限。 */
        public boolean overUpper = false;
        /** 是否超下限。 */
        public boolean underLower = false;
        /** 换算到标准量后的读数（用于与阈值比较）。 */
        public double normalizedValue = 0;
        /** 上限值（未启用为 NaN）。 */
        public double upper = Double.NaN;
        /** 下限值（未启用为 NaN）。 */
        public double lower = Double.NaN;

        public boolean isAlarm() {
            return overUpper || underLower;
        }
    }

    private ThresholdAlarm() {
    }

    /**
     * 判断读数是否超限。
     *
     * @param reading 识别结果（读数 + 单位 + 挡位）
     * @param config  阈值配置
     * @return 超限结果（未超限时 {@link Result#isAlarm()} 为 false）
     */
    public static Result check(MeterReading reading, ThresholdConfig config) {
        Result r = new Result();
        if (reading == null || config == null) {
            return r;
        }
        String category = resolveCategory(reading);
        if (category == null || category.isEmpty()) {
            return r;
        }

        double value = parseValue(reading);
        if (Double.isNaN(value)) {
            return r;
        }

        // 换算到标准量（V / A / Ω）
        String unit = reading.resolveUnit();
        double normalized = normalize(value, unit);
        if (Double.isNaN(normalized)) {
            return r;
        }

        ThresholdConfig.Bounds b = config.getBounds(category);
        if (b == null) {
            return r;
        }

        r.category = category;
        r.normalizedValue = normalized;
        if (b.upperEnabled) {
            r.upper = b.upper;
            r.overUpper = normalized > b.upper;
        }
        if (b.lowerEnabled) {
            r.lower = b.lower;
            r.underLower = normalized < b.lower;
        }
        return r;
    }

    /** 生成人类可读的报警描述文本，如 "电压超上限：36.0 V"。 */
    public static String describe(Result r, MeterReading reading) {
        if (r == null || !r.isAlarm()) {
            return "";
        }
        String unit = reading == null ? "" : reading.resolveUnit();
        String value = reading == null ? "" : reading.getDisplayText();
        StringBuilder sb = new StringBuilder();
        sb.append("⚠️ 报警：").append(r.category);
        if (r.overUpper) {
            sb.append(" 超上限 ").append(format(r.upper)).append(" ").append(unit);
        }
        if (r.underLower) {
            sb.append(" 低于下限 ").append(format(r.lower)).append(" ").append(unit);
        }
        sb.append("（当前 ").append(value).append("）");
        return sb.toString();
    }

    /** 报警语音文本，如 "电压超上限" 。 */
    public static String speechText(Result r) {
        if (r == null || !r.isAlarm()) {
            return "";
        }
        StringBuilder sb = new StringBuilder(r.category);
        if (r.overUpper) {
            sb.append("超上限");
        }
        if (r.underLower) {
            sb.append("低于下限");
        }
        sb.append("，请检查");
        return sb.toString();
    }

    /** 按挡位/单位推断量纲类别：电压/电流/电阻/电容/频率/其他。 */
    public static String resolveCategory(MeterReading reading) {
        if (reading == null) {
            return "";
        }
        String gear = reading.gear == null ? "" : reading.gear;
        String unit = reading.resolveUnit();
        if (gear.contains("电压") || gear.contains("伏") || gear.contains("V")) {
            return "电压";
        }
        if (gear.contains("电流") || gear.contains("安") || gear.contains("A")) {
            return "电流";
        }
        if (gear.contains("电阻") || gear.contains("欧") || gear.contains("Ω")) {
            return "电阻";
        }
        if (gear.contains("电容") || gear.contains("法") || (unit != null && unit.contains("F"))) {
            return "电容";
        }
        if (gear.contains("频率") || gear.contains("赫") || (unit != null && unit.contains("Hz"))) {
            return "频率";
        }
        // 只有单位时，按单位推断
        if (unit != null && !unit.isEmpty()) {
            if (unit.contains("V")) return "电压";
            if (unit.contains("A")) return "电流";
            if (unit.contains("Ω")) return "电阻";
            if (unit.contains("F")) return "电容";
            if (unit.contains("Hz")) return "频率";
        }
        return "";
    }

    /** 解析数值，无法解析返回 NaN。 */
    private static double parseValue(MeterReading reading) {
        String v = reading.value == null ? "" : reading.value.trim();
        if (v.isEmpty()) {
            return Double.NaN;
        }
        try {
            return Double.parseDouble(v);
        } catch (Exception e) {
            return Double.NaN;
        }
    }

    /** 把带单位的数值换算到标准量：V/A/Ω/F/Hz；无法换算返回 NaN。 */
    private static double normalize(double value, String unit) {
        if (unit == null) {
            return value;
        }
        String u = unit.trim();
        switch (u) {
            case "mV": return value * 1e-3;
            case "kV": return value * 1e3;
            case "mA": return value * 1e-3;
            case "uA":
            case "μA": return value * 1e-6;
            case "kΩ": return value * 1e3;
            case "MΩ": return value * 1e6;
            case "mF": return value * 1e-3;
            case "uF":
            case "μF": return value * 1e-6;
            case "kHz": return value * 1e3;
            case "MHz": return value * 1e6;
            case "V":
            case "A":
            case "Ω":
            case "F":
            case "Hz":
            case "℃":
            case "°C": return value;
            default: return value;
        }
    }

    private static String format(double d) {
        if (d == (long) d) {
            return String.valueOf((long) d);
        }
        return String.valueOf(d);
    }
}
