package com.ar.glass.core;

import android.util.Log;

import com.xy.ksdk.cmd.api.XyCmd;
import com.xy.ksdk.cmd.base.Cmd;
import com.xy.ksdk.cmd.base.XyCmdListener;
import com.xy.ksdk.protos.FileBody;

import org.json.JSONObject;

import java.io.File;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

/**
 * BLE 直传文件接收器（最稳定的回传路径：无需 WiFi/P2P，照片经 BLE 串口通道分块推送）。
 *
 * 协议（反编译 ksdk-release.aar 确认）：
 *   帧:   ##[type][len低][len高][data...]$$   —— XyCmd 内部自动找帧头/组包/去帧头
 *   type='0'(48): JSON 命令 {"C":"cs_xxx","W":1,"B":"{...}"}
 *   type=其他:    FileMessage protobuf 文件分块
 *         字段: filename/fileType/fileSize/packSize/packIndex/tag/dataLen/data
 *
 * 手机侧命令（ksdk CmdSendManager 同名命令）：
 *   cs_sdfl               请求眼镜文件列表
 *   cs_asfl {ftype}       请求眼镜推送文件（ftype 实测确定，先按 1=照片）
 *   cs_flrv {ftype,fname} 告知眼镜开始接收某文件
 *   cs_flts {state,index} 传输状态/收块确认（收完一包回一次）
 *
 * 眼镜推送的分块按 packIndex*packSize 偏移拼装写入临时文件；
 * 收满 fileSize 字节判定完成 → 移交 photos 目录 → 回调 onFileReady。
 */
public class BleFileTransport implements XyCmdListener {

    private static final String TAG = "BleFileTransport";
    /** cmdType='0' 为 JSON 命令帧 */
    private static final byte TYPE_JSON = 48;

    /** 上层回调（GlassBleService 注入，运行于调用线程） */
    public interface Callback {
        void onLog(String msg);

        void onFileReady(File file);

        /** 传输进度（percent 0-100；info 为简短状态文本，供进度条显示） */
        default void onProgress(int percent, String info) {}
    }

    private final Callback mCb;
    private final XyCmd mXyCmd = new XyCmd(this);

    // ===== 接收状态（同一时刻只收一个文件） =====
    private String mFileName;
    private int mFileSize;
    private int mPackSize;
    private int mReceived;
    private int mLastIndex;
    private RandomAccessFile mRaf;
    private File mOutFile;

    /** 文件块帧的 cmdType 取值（眼镜固件决定，'0' 是 JSON；实测后可在 ADB 命令中修正） */
    private volatile int mFileType = -1;

    public BleFileTransport(Callback cb) {
        mCb = cb;
    }

    /** 喂入 BLE 通知通道原始数据（串口/NUS 通道都可喂；0xBC 帧会被帧头查找跳过，安全） */
    public void feed(byte[] data) {
        if (data == null || data.length == 0) return;
        try {
            mXyCmd.parse(data, 0, data.length);
        } catch (Throwable e) {
            Log.e(TAG, "parse error", e);
        }
    }

    @Override
    public void onCmdRecv(byte type, byte[] data, int off, int len) {
        int t = type & 0xFF;
        if (t == TYPE_JSON) {
            handleJson(data, off, len);
            return;
        }
        // 文件块帧
        if (mFileType == -1) {
            mFileType = t;
            mCb.onLog("🧩 识别到文件块帧 type=" + t);
        }
        if (mFileType != t) return;
        try {
            byte[] raw = Arrays.copyOfRange(data, off, off + len);
            handleFileMessage(FileBody.FileMessage.parseFrom(raw));
        } catch (Throwable e) {
            Log.e(TAG, "file block parse error", e);
            mCb.onLog("⚠️ 文件块解析失败 type=" + t + " len=" + len);
        }
    }

    // ===== JSON 命令 =====

    private void handleJson(byte[] data, int off, int len) {
        try {
            String json = new String(data, off, len, StandardCharsets.UTF_8);
            JSONObject o = new JSONObject(json);
            Cmd cmd = new Cmd();
            cmd.cmd = o.optString("C");
            cmd.body = o.optString("B");
            mCb.onLog("📩 [BLE命令] " + cmd.cmd + " body=" + cmd.body);
        } catch (Throwable e) {
            Log.e(TAG, "json parse error", e);
        }
    }

    // ===== 文件块拼装 =====

    private void handleFileMessage(FileBody.FileMessage msg) {
        String name = msg.getFilename();
        if (name == null || name.isEmpty()) name = "ble_photo_" + System.currentTimeMillis() + ".jpg";
        // 新文件开始：初始化接收状态
        if (!name.equals(mFileName)) {
            closeQuietly();
            mFileName = name;
            mFileSize = msg.getFileSize();
            mPackSize = msg.getPackSize() > 0 ? msg.getPackSize() : 512;
            mReceived = 0;
            mLastIndex = -1;
            mOutFile = new File(AppState.getInstance().getAppContext()
                    .getExternalFilesDir(null), "glass_media/tmp/" + name);
            //noinspection ResultOfMethodCallIgnored
            mOutFile.getParentFile().mkdirs();
            try {
                // 预分配，支持乱序按偏移写
                mRaf = new RandomAccessFile(mOutFile, "rws");
                mRaf.setLength(Math.max(mFileSize, 1));
            } catch (Exception e) {
                mCb.onLog("⚠️ BLE收文件: 创建失败 " + e.getMessage());
                closeQuietly();
                return;
            }
            mCb.onLog("📥 BLE收文件开始: " + name + " 大小=" + mFileSize + "B 包=" + mPackSize + "B");
            mCb.onProgress(0, "BLE 回传开始");
        }

        int index = msg.getPackIndex();
        byte[] data = msg.getData() != null ? msg.getData().toByteArray() : new byte[0];
        long offset = (long) Math.max(index, 0) * mPackSize;
        try {
            mRaf.seek(offset);
            mRaf.write(data);
        } catch (Exception e) {
            mCb.onLog("⚠️ BLE收文件: 写入失败 idx=" + index + " " + e.getMessage());
            return;
        }
        mReceived += data.length;
        mLastIndex = index;
        if (mFileSize > 0) {
            int percent = (int) Math.min(100L, mReceived * 100L / mFileSize);
            mCb.onProgress(percent, String.format("BLE 回传 %d%% (%s/%s)",
                    percent, formatSize(mReceived), formatSize(mFileSize)));
        }
        // 每包确认（cs_flts），state 语义未文档化，先回 state=0 表示继续
        // notifyTransferState(0, index);

        if (mReceived >= mFileSize) {
            // 收完：关闭并移交 photos 目录
            closeQuietly();
            File dst = new File(mOutFile.getParentFile().getParentFile(), "photos/" + mFileName);
            //noinspection ResultOfMethodCallIgnored
            dst.getParentFile().mkdirs();
            if (mOutFile.renameTo(dst) || mOutFile.exists()) {
                mCb.onLog("✅ BLE收文件完成: " + dst.getName());
                // 收完确认 cs_flts state=1
                notifyTransferState(1, mLastIndex);
                mFileName = null;
                mCb.onFileReady(dst);
            } else {
                mCb.onLog("⚠️ BLE收文件: 移交失败");
            }
        }
    }

    // ===== 对眼镜的命令（直接复用 ksdk 命令类，写入桥由 GlassBleService 注入） =====

    /** ksdk 命令输出桥：写入 BLE 串口队列 */
    public interface Writer {
        void write(byte[] data);
    }

    private volatile Writer mWriter;

    public void setWriter(Writer w) {
        mWriter = w;
    }

    private boolean send(com.xy.ksdk.cmd.base.SCmd cmd) {
        Writer w = mWriter;
        if (w == null) {
            mCb.onLog("⚠️ BLE命令发送失败: 串口未就绪");
            return false;
        }
        return cmd.send(new com.xy.ksdk.api.cmd.IBluetooth() {
            @Override
            public void write(byte[] data) {
                w.write(data);
            }
        });
    }

    /** 请求眼镜推送文件列表 */
    public void requestFileList() {
        send(new com.xy.ksdk.cmd.cmd.S_Sdfl());
        mCb.onLog("📤 已请求文件列表 cs_sdfl");
    }

    /** 请求眼镜推送文件（ftype 先按 1=照片 实测） */
    public void requestFile(int ftype) {
        com.xy.ksdk.cmd.cmd.S_AskFile c = new com.xy.ksdk.cmd.cmd.S_AskFile();
        c.setFileType((byte) ftype);
        send(c);
        mCb.onLog("📤 已请求文件 cs_asfl ftype=" + ftype);
    }

    /** 告知眼镜开始接收某文件 */
    public void ackRecv(int ftype, String fname) {
        com.xy.ksdk.cmd.cmd.S_FileRecv c = new com.xy.ksdk.cmd.cmd.S_FileRecv();
        c.setFileType((byte) ftype);
        c.setFileName(fname);
        send(c);
        mCb.onLog("📤 已确认接收 cs_flrv ftype=" + ftype + " fname=" + fname);
    }

    /** 传输状态上报（state 语义未文档化：0=继续收，1=收完确认） */
    public void notifyTransferState(int state, int index) {
        com.xy.ksdk.cmd.cmd.S_AIImageFile c = new com.xy.ksdk.cmd.cmd.S_AIImageFile();
        c.setData(state, index);
        send(c);
        mCb.onLog("📤 已上报传输状态 cs_flts state=" + state + " index=" + index);
    }

    private void closeQuietly() {
        try {
            if (mRaf != null) mRaf.close();
        } catch (Exception ignored) {
        }
        mRaf = null;
    }

    private static String formatSize(int bytes) {
        return bytes >= 1024 * 1024
                ? String.format("%.1fMB", bytes / 1048576.0)
                : String.format("%dKB", bytes / 1024);
    }

    /** 调试用：当前是否有文件在接收 */
    public boolean isReceiving() {
        return mFileName != null;
    }

    /** 调试用：接收进度描述 */
    public String progress() {
        return mFileName == null ? "空闲" : mFileName + " " + mReceived + "/" + mFileSize
                + "B 文件块type=" + mFileType;
    }
}
