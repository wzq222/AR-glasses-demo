package com.ar.glass.util;

/**
 * EventBus 事件消息类
 * 用于在不同组件间传递消息（蓝牙状态、指令结果等）
 */
public class EventMsg {

    public static final int MSG_CONNECT_STATE = 1;
    public static final int MSG_SYSTEM_READY = 3;
    public static final int MSG_WIFI_CONNECT_RESULT = 9;
    public static final int MSG_FILE_RECV_FINISH = 11;
    public static final int MSG_TOAST = 100;
    public static final int MSG_SYNC_COMPLETE = 102;      // 文件同步完成
    public static final int MSG_LOG = 103;                // 仅追加到运行日志（不弹Toast）
    public static final int MSG_PHOTO_LIST = 105;          // 眼镜照片列表已获取（obj=List<String>）
    public static final int MSG_QR_RESULT = 106;           // 二维码识别结果（obj=String，空串表示未识别到）
    public static final int MSG_BATTERY_UPDATE = 107;       // 眼镜电量更新（arg1=电量0-100，arg2=1表示充电中）

    public EventMsg(int what, int arg1, int arg2) {
        this.what = what;
        this.arg1 = arg1;
        this.arg2 = arg2;
    }

    public int what;
    public int arg1;
    public int arg2;
    public Object obj;

    public EventMsg(int what) {
        this.what = what;
    }

    public EventMsg(int what, int arg1) {
        this.what = what;
        this.arg1 = arg1;
    }

    public EventMsg(int what, Object obj) {
        this.what = what;
        this.obj = obj;
    }

    public EventMsg(int what, int arg1, Object obj) {
        this.what = what;
        this.arg1 = arg1;
        this.obj = obj;
    }
}
