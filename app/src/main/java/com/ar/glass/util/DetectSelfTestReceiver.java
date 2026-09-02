package com.ar.glass.util;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import com.ar.glass.util.EventMsg;
import com.ar.glass.vision.DetectResult;
import com.ar.glass.vision.MarkedPointDetectorHolder;
import com.ar.glass.vision.YoloDetector;

import org.greenrobot.eventbus.EventBus;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * YOLO 检测自测入口（无需眼镜硬件）：
 *   adb shell am broadcast -a com.ar.glass.DETECT_SELF_TEST
 * 对 glass_media/photos 中最新一张照片执行 YOLO 检测，
 * 结果经 MSG_DETECT_RESULT 事件走与真实检测循环完全相同的 UI/播报路径。
 * 用于模拟器/真机上在未连接眼镜时验证推理链路。
 */
public class DetectSelfTestReceiver extends BroadcastReceiver {

    private static final String TAG = "DetectSelfTest";

    /** 供 ADB 调试接口（GlassDebugReceiver）直接触发本机照片检测 */
    public static void sendDetect(Context context) {
        new DetectSelfTestReceiver().onReceive(context.getApplicationContext(), new Intent());
    }

    @Override
    public void onReceive(Context context, Intent intent) {
        // 检索顺序：外部 glass_media/photos（真实同步路径）→ 内部 files/glass_media/photos（自测推入路径）
        File latest = findLatest(new File(context.getExternalFilesDir(null), "glass_media/photos"));
        if (latest == null) {
            latest = findLatest(new File(context.getFilesDir(), "glass_media/photos"));
        }
        final File photo = latest;
        postLog("自测: 最新照片 = " + (photo != null ? photo.getAbsolutePath() : "无"));
        if (photo == null) {
            EventBus.getDefault().post(new EventMsg(EventMsg.MSG_DETECT_RESULT, new DetectResult("无照片")));
            return;
        }
        final Context app = context.getApplicationContext();
        new Thread(() -> {
            try {
                if (!MarkedPointDetectorHolder.isReady(app)) {
                    String error = MarkedPointDetectorHolder.getInitializationError();
                    postLog("自测: 防松标记模型未就绪: " + error);
                    EventBus.getDefault().post(new EventMsg(EventMsg.MSG_DETECT_RESULT,
                            new DetectResult(error == null ? "防松标记模型加载失败" : error)));
                    return;
                }
                BitmapFactory.Options opts = new BitmapFactory.Options();
                opts.inJustDecodeBounds = true;
                BitmapFactory.decodeFile(photo.getAbsolutePath(), opts);
                int sample = 1;
                while (Math.max(opts.outWidth, opts.outHeight) / (sample * 2) >= 1280) sample *= 2;
                opts.inSampleSize = sample;
                opts.inJustDecodeBounds = false;
                Bitmap bmp = BitmapFactory.decodeFile(photo.getAbsolutePath(), opts);
                if (bmp == null) {
                    EventBus.getDefault().post(new EventMsg(EventMsg.MSG_DETECT_RESULT, new DetectResult("照片解码失败")));
                    return;
                }
                MarkedPointDetectorHolder.Result marked = MarkedPointDetectorHolder.detect(app, bmp);
                List<YoloDetector.Detection> dets = marked.detections;
                long ms = Math.round(marked.latencyMillis);
                postLog("自测: " + dets.size() + " 个防松标记检查点, 推理 " + ms + "ms");
                for (YoloDetector.Detection d : dets) {
                    Log.i(TAG, String.format("det %s %.2f (%.3f,%.3f,%.3f,%.3f)",
                            d.className, d.score, d.x1, d.y1, d.x2, d.y2));
                }
                // bmp 所有权交给 UI 作为预览图
                EventBus.getDefault().post(new EventMsg(EventMsg.MSG_DETECT_RESULT, dets.size(),
                        new DetectResult(bmp, dets, bmp.getWidth(), bmp.getHeight(), ms, photo.getName())));
            } catch (Throwable e) {
                Log.e(TAG, "self test failed", e);
                postLog("自测异常: " + e.getMessage());
                EventBus.getDefault().post(new EventMsg(EventMsg.MSG_DETECT_RESULT, new DetectResult(e.getMessage())));
            }
        }).start();
    }

    /** 在目录中找最新的图片文件；目录不存在或为空返回 null */
    private static File findLatest(File dir) {
        File[] files = dir.listFiles();
        File latest = null;
        long max = 0;
        if (files != null) {
            for (File f : files) {
                String n = f.getName().toLowerCase();
                if (f.isFile() && (n.endsWith(".jpg") || n.endsWith(".jpeg") || n.endsWith(".png"))
                        && f.lastModified() > max) {
                    max = f.lastModified();
                    latest = f;
                }
            }
        }
        return latest;
    }

    private static void postLog(String text) {
        Log.i(TAG, text);
        EventBus.getDefault().post(new EventMsg(EventMsg.MSG_LOG, "🧪 " + text));
    }
}
