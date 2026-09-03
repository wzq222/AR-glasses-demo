package com.ar.glass.core;

import android.content.Context;

/**
 * 全局状态管理类（单例模式）
 * 保存应用运行时的全局状态，类似 demo 中的 CC.java
 */
public class AppState {

    private static AppState instance;

    private Context appContext;

    public boolean isBleConnected = false;
    public boolean isSystemReady = false;
    public boolean isSocketConnected = false;

    /** 眼镜电量（0-100），-1 表示未知 */
    public int batteryLevel = -1;
    /** 眼镜是否充电中 */
    public boolean isCharging = false;

    public String bleAddress;
    public String bleName;

    public String serverIp = "";

    private AppState() {}

    public static AppState getInstance() {
        if (instance == null) {
            synchronized (AppState.class) {
                if (instance == null) {
                    instance = new AppState();
                }
            }
        }
        return instance;
    }

    public void init(Context context) {
        appContext = context.getApplicationContext();
    }

    public Context getAppContext() {
        return appContext;
    }

    public void resetConnection() {
        isBleConnected = false;
        isSystemReady = false;
        isSocketConnected = false;
    }
}
