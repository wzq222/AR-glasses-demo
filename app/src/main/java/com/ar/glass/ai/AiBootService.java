package com.ar.glass.ai;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Intent;
import android.os.Build;
import android.os.IBinder;

/**
 * AI 常驻前台服务：让进程在后台存活，使已加载模型与本地 API 服务持续可用。
 *
 * <p>服务本身很薄——只负责 {@code startForeground} + 拉起 {@link AiRuntime#autoStart}。
 * 真正的加载/卸载逻辑都在 AiRuntime 内。
 */
public class AiBootService extends Service {

    private static final String CHANNEL_ID = "ai_runtime";
    private static final int NOTIF_ID = 0x1A1;

    @Override
    public void onCreate() {
        super.onCreate();
        try {
            startForeground(NOTIF_ID, buildNotification());
        } catch (Throwable t) {
            AiDebug.e("AiBootService: startForeground failed: " + t);
        }
        AiRuntime.get().autoStart(this);
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        AiRuntime.get().autoStart(this);
        return START_STICKY;
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    private Notification buildNotification() {
        NotificationManager nm = (NotificationManager) getSystemService(NOTIFICATION_SERVICE);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O && nm != null
                && nm.getNotificationChannel(CHANNEL_ID) == null) {
            NotificationChannel ch = new NotificationChannel(CHANNEL_ID, "AI 运行时",
                    NotificationManager.IMPORTANCE_MIN);
            ch.setShowBadge(false);
            nm.createNotificationChannel(ch);
        }
        Notification.Builder b = (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
                ? new Notification.Builder(this, CHANNEL_ID)
                : new Notification.Builder(this);
        return b.setContentTitle("AR 眼镜 · AI 运行时")
                .setContentText("本地 ASR / LLM 与 API 服务运行中")
                .setSmallIcon(android.R.drawable.stat_notify_sync)
                .setOngoing(true)
                .build();
    }
}
