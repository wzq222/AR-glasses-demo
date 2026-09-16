package com.ar.glass.core;

import android.app.Application;
import android.content.Intent;
import android.util.Log;

import com.ar.glass.ai.AiBootService;
import com.ar.glass.ai.AiRuntime;
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
        startAiRuntime();
    }

    /**
     * 启动 AI 常驻运行时：前台服务保活 + 自动加载模型 + 开启本地 API（默认开）。
     * 任何异常都不影响主流程（AI 为可选能力）。
     */
    private void startAiRuntime() {
        try {
            AiRuntime.get().install(this);
            // 直接触发模型加载与本地 API 启动：**不依赖前台服务**。
            // （Android 12+ 禁止从 Application 后台启动前台服务，靠服务触发会失效。）
            AiRuntime.get().autoStart(this);
            Log.d(TAG, "AiRuntime autoStart issued");
        } catch (Throwable t) {
            Log.e(TAG, "AiRuntime autoStart failed", t);
        }
        // 前台服务仅用于保活：从 Application 启动可能被系统拒绝，
        // 失败不影响加载（MainActivity 在前台上下文会再试一次）。
        try {
            Intent i = new Intent(this, AiBootService.class);
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                startForegroundService(i);
            } else {
                startService(i);
            }
        } catch (Throwable t) {
            Log.w(TAG, "start AiBootService rejected (will retry from Activity)", t);
        }
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
