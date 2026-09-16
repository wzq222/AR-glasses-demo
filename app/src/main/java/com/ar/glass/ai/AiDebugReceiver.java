package com.ar.glass.ai;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;

/**
 * AI 调试广播（adb 直驱，模拟器/自动化干跑调试用）：
 *
 * <pre>
 * // 注入测试图（作为下一条发送消息的附带图）
 * am broadcast -n com.ar.glass/.ai.AiDebugReceiver -a com.ar.glass.AI_DEBUG \
 *     --es cmd img --es path /sdcard/Pictures/test.png
 *
 * // 运行时状态 / 内存快照 / 连接信息（可复制给调用方）
 *   --es cmd status
 *   --es cmd mem
 *   --es cmd apiinfo
 *
 * // 加载 / 卸载模型（卸载用于性能对比基线）
 *   --es cmd load      [--es path /sdcard/xxx.gguf]
 *   --es cmd unload
 *
 * // 本地 API 开关
 *   --es cmd api --es arg on|off
 *
 * // 干跑生成（时延 / 估算 tok/s）
 *   --es cmd gen --es text "你好" [--ei max 64]
 * </pre>
 */
public class AiDebugReceiver extends BroadcastReceiver {

    @Override
    public void onReceive(Context context, Intent intent) {
        if (!"com.ar.glass.AI_DEBUG".equals(intent.getAction())) return;
        String cmd = intent.getStringExtra("cmd");
        if (cmd == null) cmd = "";
        AiRuntime rt = AiRuntime.get();
        rt.install(context);
        AiDebug.i("aidbg: cmd=" + cmd);

        switch (cmd) {
            case "img":
                handleImage(intent);
                break;

            case "status":
                AiDebug.i("runtime.status: " + rt.status());
                AiDebug.i("runtime.mem: " + rt.memStatus());
                break;

            case "mem":
                AiDebug.i("runtime.mem: " + rt.memStatus());
                break;

            case "apiinfo":
                AiDebug.i("runtime.apiinfo:\n" + rt.connInfo());
                break;

            case "load": {
                String path = intent.getStringExtra("path");
                rt.loadAsync(path);
                AiDebug.i("runtime: load requested (path="
                        + (path == null || path.isEmpty() ? "内置" : path) + ")");
                break;
            }

            case "unload":
                rt.unload("adb");
                break;

            case "api": {
                String arg = intent.getStringExtra("arg");
                boolean on = arg == null || !"off".equalsIgnoreCase(arg.trim());
                if (arg == null) on = intent.getBooleanExtra("on", true);
                rt.setApiEnabled(on);
                AiDebug.i("runtime: api " + (on ? "on" : "off") + " requested");
                break;
            }

            case "gen": {
                String text = intent.getStringExtra("text");
                int max = intent.getIntExtra("max", 64);
                rt.dryRunGen(text, max);
                break;
            }

            case "cancel":
                rt.cancelGen();
                AiDebug.i("runtime: cancel requested (adb)");
                break;

            default:
                AiDebug.e("aidbg: unknown cmd=" + cmd);
        }
    }

    private void handleImage(Intent intent) {
        String path = intent.getStringExtra("path");
        if (path == null || path.isEmpty()) {
            AiDebug.e("aidbg: path missing");
            return;
        }
        try {
            Bitmap bm = android.graphics.ImageDecoder.decodeBitmap(
                    android.graphics.ImageDecoder.createSource(
                            new java.io.File(path)),
                    (decoder, info, src) -> {
                        int maxSide = 896;
                        int w = info.getSize().getWidth();
                        int h = info.getSize().getHeight();
                        float scale = Math.min(1f,
                                maxSide / (float) Math.max(w, h));
                        if (scale < 1f) {
                            decoder.setTargetSize(
                                    Math.max(1, (int) (w * scale)),
                                    Math.max(1, (int) (h * scale)));
                        }
                    });
            AiDebugBus.pendingImage = bm;
            AiDebug.i("aidbg: image loaded " + bm.getWidth() + "x" + bm.getHeight()
                    + " from " + path);
        } catch (Exception e) {
            AiDebug.e("aidbg: decode failed " + e);
        }
    }
}
