package com.ar.glass.ai;

import android.graphics.Bitmap;

/** 进程内调试数据总线：调试广播注入的待发送图片等。 */
public final class AiDebugBus {
    public static volatile Bitmap pendingImage;

    private AiDebugBus() {
    }
}
