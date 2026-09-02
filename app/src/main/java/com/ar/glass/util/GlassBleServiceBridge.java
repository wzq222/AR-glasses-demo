package com.ar.glass.util;

import android.util.Log;

import com.ar.glass.core.GlassBleService;

/**
 * ADB 调试命令 → GlassBleService 的静态桥（薄委托，服务未就绪时仅打日志）。
 */
public final class GlassBleServiceBridge {

    private static final String TAG = "GlassDebug";

    private GlassBleServiceBridge() {
    }

    public static void writeRaw(byte[] data) {
        if (GlassBleService.debugInstance() != null) {
            GlassBleService.debugWriteRaw(data);
        } else {
            Log.w(TAG, "writeRaw: 服务未就绪");
        }
    }

    public static void takePhoto() {
        if (GlassBleService.debugInstance() != null) {
            GlassBleService.debugTakePhoto();
        } else {
            Log.w(TAG, "takePhoto: 服务未就绪");
        }
    }

    public static void startSync() {
        if (GlassBleService.debugInstance() != null) {
            GlassBleService.debugStartSync();
        } else {
            Log.w(TAG, "startSync: 服务未就绪");
        }
    }

    public static void stopHotspot() {
        if (GlassBleService.debugInstance() != null) {
            GlassBleService.debugStopHotspot();
        } else {
            Log.w(TAG, "stopHotspot: 服务未就绪");
        }
    }

    /** 拍照后经 BLE 直传拉取照片 */
    public static void bleGet() {
        if (GlassBleService.debugInstance() != null) {
            GlassBleService.debugBleGet();
        } else {
            Log.w(TAG, "bleGet: 服务未就绪");
        }
    }

    /** 请求眼镜文件列表 */
    public static void bleList() {
        if (GlassBleService.debugInstance() != null) {
            GlassBleService.debugBleList();
        } else {
            Log.w(TAG, "bleList: 服务未就绪");
        }
    }

    /** BLE 文件接收进度 */
    public static String bleProgress() {
        return GlassBleService.debugBleProgress();
    }

    /** 手动注入眼镜 IP 并触发照片列表拉取（模拟器/无蓝牙调试场景） */
    public static void setGlassesIp(String ip) {
        GlassBleService.debugSetGlassesIp(ip);
    }

    /** BLE 直拉眼镜最近照片缩略图（oudmon cmd=0xFD，落地后本地检测） */
    public static void thumbGet() {
        GlassBleService.debugThumbGet();
    }

    /** YOLO 模式拍照（回传落地后自动本地检测） */
    public static void takePhotoDetect() {
        GlassBleService.debugTakePhotoDetect();
    }

    /** 直连上次成功连接的眼镜（跳过设备选择），返回结果描述 */
    public static String reconnectLast() {
        return GlassBleService.debugReconnectLast();
    }

    /** 全量回传：拉取眼镜全部照片并自动导入原图库 */
    public static String syncAll() {
        return GlassBleService.debugSyncAll();
    }

    /** 清空原图库（UI 需二次确认），返回删除数量 */
    public static int clearPhotos() {
        return GlassBleService.debugClearPhotos();
    }

    /** 当前 BLE 写队列状态 */
    public static String queueStatus() {
        return GlassBleService.debugQueueStatus();
    }
}
