package com.ar.glass.ai;

import android.os.Build;
import android.util.Log;

/**
 * AI 调试日志：统一输出到 AiLog 标签，adb logcat -s AiLog AiLlm AiAsr 直接读取。
 * 首次访问时打印一份设备/内存快照，便于远程定位后端选择问题。
 */
public final class AiDebug {
    private static final String TAG = "AiLog";
    private static boolean dumped = false;

    private AiDebug() {
    }

    public static synchronized void deviceSnapshot() {
        if (dumped) return;
        dumped = true;
        Runtime rt = Runtime.getRuntime();
        long memTotal = 0, memAvail = 0;
        try {
            java.io.BufferedReader r = new java.io.BufferedReader(
                    new java.io.FileReader("/proc/meminfo"));
            String line;
            while ((line = r.readLine()) != null) {
                String[] p = line.split("\\s+");
                if (line.startsWith("MemTotal")) memTotal = Long.parseLong(p[1]);
                if (line.startsWith("MemAvailable")) memAvail = Long.parseLong(p[1]);
            }
            r.close();
        } catch (Exception ignored) {
        }
        Log.i(TAG, "device: " + Build.MANUFACTURER + " " + Build.MODEL
                + ", android " + Build.VERSION.RELEASE
                + ", abi=" + Build.SUPPORTED_ABIS[0]
                + ", cores=" + rt.availableProcessors()
                + String.format(", mem %.1f/%.1f GB",
                memAvail / 1048576.0, memTotal / 1048576.0));
    }

    public static void i(String msg) {
        Log.i(TAG, msg);
    }

    public static void e(String msg) {
        Log.e(TAG, msg);
    }
}
