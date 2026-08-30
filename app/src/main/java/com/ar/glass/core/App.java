package com.ar.glass.core;

import android.app.Application;
import android.util.Log;

import com.iflytek.sparkchain.core.SparkChain;
import com.iflytek.sparkchain.core.SparkChainConfig;

/**
 * 应用入口 Application 类
 * 在这里做全局初始化
 */
public class App extends Application {

    private static final String TAG = "App";

    // ===== 讯飞 SparkChain 三元组（语音听写）=====
    private static final String SPARK_APP_ID = "8e7e02af";
    private static final String SPARK_API_KEY = "c9b7d9e146864ffbeb94d6ea2d379b37";
    private static final String SPARK_API_SECRET = "NGJkMDAwYzg5YzI0NmQ5YmU4MDY3ZWEx";

    @Override
    public void onCreate() {
        super.onCreate();
        AppState.getInstance().init(this);
        initSparkChain();
    }

    private void initSparkChain() {
        try {
            SparkChainConfig config = SparkChainConfig.builder();
            config.appID(SPARK_APP_ID)
                    .apiKey(SPARK_API_KEY)
                    .apiSecret(SPARK_API_SECRET);
            int ret = SparkChain.getInst().init(this, config);
            Log.d(TAG, "SparkChain init result: " + ret);
        } catch (Throwable t) {
            Log.e(TAG, "SparkChain init error", t);
        }
    }
}
