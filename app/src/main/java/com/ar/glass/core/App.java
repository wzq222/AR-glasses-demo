package com.ar.glass.core;

import android.app.Application;

/**
 * 应用入口 Application 类
 * 在这里做全局初始化
 */
public class App extends Application {

    @Override
    public void onCreate() {
        super.onCreate();
        AppState.getInstance().init(this);
    }
}
