package com.ar.glass.vision;

/**
 * 万用表结构化识别结果。
 *
 * 由云端视觉大模型一次性识别出：屏幕读数、单位符号、旋钮挡位、异常提示，
 * 供台账保存、语音播报、量纲补全、异常告警等下游使用。
 */
public class MeterReading {

    /** 屏幕数字读数，如 "24.8"（不含单位）。 */
    public String value = "";

    /** 屏幕上显示的单位符号，如 V/mV/A/mA/Ω/kΩ/MΩ；无则空串。 */
    public String unit = "";

    /** 旋钮所在挡位，如 "直流电压"/"交流电压"/"电阻"/"通断蜂鸣"等；无则空串。 */
    public String gear = "";

    /** 异常提示，如 "表笔插孔与挡位不符"；无异常则空串。 */
    public String warning = "";

    /** 模型原始输出文本（排查用）。 */
    public String raw = "";

    public boolean hasValue() {
        return value != null && !value.trim().isEmpty();
    }

    /** 读数 + 单位，如 "24.8 V"；单位缺失但可推断时也补全。 */
    public String getDisplayText() {
        String v = value == null ? "" : value.trim();
        if (v.isEmpty()) {
            return "";
        }
        String u = resolveUnit();
        return u.isEmpty() ? v : v + " " + u;
    }

    /** 语音播报文本：单位转中文，如 "24.8 伏"。 */
    public String getSpeechText() {
        String v = value == null ? "" : value.trim();
        if (v.isEmpty()) {
            return "";
        }
        String u = resolveUnit();
        String num = v.replace("-", "负");
        return u.isEmpty() ? num : num + " " + unitToChinese(u);
    }

    /** 解析后的数值（尽力而为，用于阈值判断等场景）。 */
    public double parseValue(double fallback) {
        try {
            return Double.parseDouble(value == null ? "" : value.trim());
        } catch (Exception e) {
            return fallback;
        }
    }

    /**
     * 量纲补全：优先用屏幕识别到的单位；若缺失但识别出挡位，按挡位推断单位。
     */
    public String resolveUnit() {
        String u = unit == null ? "" : unit.trim();
        if (!u.isEmpty()) {
            return u;
        }
        return inferUnitFromGear();
    }

    /** 按挡位推断单位（直流/交流电压→V，电流→A，电阻→Ω 等）。 */
    public String inferUnitFromGear() {
        String g = gear == null ? "" : gear;
        if (g.contains("电压") || g.contains("伏") || g.contains("V")) {
            return "V";
        }
        if (g.contains("电流") || g.contains("安") || g.contains("A")) {
            return "A";
        }
        if (g.contains("电阻") || g.contains("欧") || g.contains("Ω")) {
            return "Ω";
        }
        if (g.contains("电容") || g.contains("法")) {
            return "F";
        }
        if (g.contains("频率") || g.contains("赫")) {
            return "Hz";
        }
        return "";
    }

    private static String unitToChinese(String unit) {
        if (unit == null) {
            return "";
        }
        switch (unit) {
            case "V":  return "伏";
            case "mV": return "毫伏";
            case "kV": return "千伏";
            case "A":  return "安";
            case "mA": return "毫安";
            case "uA":
            case "μA": return "微安";
            case "Ω":  return "欧姆";
            case "kΩ": return "千欧";
            case "MΩ": return "兆欧";
            case "Hz": return "赫兹";
            case "kHz": return "千赫";
            case "℃":
            case "°C": return "摄氏度";
            default: return unit;
        }
    }
}
