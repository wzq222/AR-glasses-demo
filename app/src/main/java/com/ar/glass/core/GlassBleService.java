package com.ar.glass.core;

import android.annotation.SuppressLint;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothGatt;
import android.bluetooth.BluetoothGattCallback;
import android.bluetooth.BluetoothGattCharacteristic;
import android.bluetooth.BluetoothGattDescriptor;
import android.bluetooth.BluetoothGattService;
import android.bluetooth.BluetoothManager;
import android.bluetooth.BluetoothProfile;
import android.bluetooth.le.BluetoothLeScanner;
import android.bluetooth.le.ScanCallback;
import android.bluetooth.le.ScanResult;
import android.bluetooth.le.ScanSettings;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.wifi.WifiConfiguration;
import android.net.wifi.WifiManager;
import android.net.wifi.p2p.WifiP2pConfig;
import android.net.wifi.p2p.WifiP2pDevice;
import android.net.wifi.p2p.WifiP2pInfo;
import android.net.wifi.p2p.WifiP2pManager;
import android.os.Binder;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.os.Message;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.core.app.NotificationCompat;
import androidx.core.app.ServiceCompat;
import androidx.core.content.ContextCompat;

import com.ar.glass.R;
import com.ar.glass.ui.MainActivity;
import com.ar.glass.util.EventMsg;
import com.ar.glass.vision.DetectResult;
import com.ar.glass.vision.MarkedPointDetectorHolder;
import com.ar.glass.vision.Vision;
import com.ar.glass.vision.YoloDetector;

import org.greenrobot.eventbus.EventBus;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.lang.reflect.Method;

/**
 * 眼镜蓝牙服务 - CY01 原生 BLE 版本
 *
 * 背景：K900 SDK 的协议(UUID 00004860...)与 CY01 眼镜完全不匹配，导致连接成功但
 * 永远收不到数据（一直"系统未就绪"）。经协议调试确认 CY01 使用
 * 两套 BLE 服务：
 *
 *   1) NUS 控制通道（Nordic UART Service）
 *      - 服务:   6e40fff0-b5a3-f393-e0a9-e50e24dcca9e
 *      - 写特征: 6e400002-b5a3-f393-e0a9-e50e24dcca9e
 *      - 通知特征: 6e400003-b5a3-f393-e0a9-e50e24dcca9e
 *
 *   2) 串口数据/文件通道（自定义）
 *      - 服务:   de5bf728-d711-4e47-af26-65e3012a5dc7
 *      - 通知:  de5bf729-d711-4e47-af26-65e3012a5dc7
 *      - 写:    de5bf72a-d711-4e47-af26-65e3012a5dc7
 *
 * 串口大数据帧格式：
 *   [0xBC][action][len低][len高][CRC16低][CRC16高][payload...]
 *   空 payload 时长度区为 0x0000、CRC 区为 0xFFFF。
 *
 * 本阶段目标：打通数据通道 —— 连接、订阅通知、发送初始化/心跳命令，确认能收到眼镜数据。
 */
@SuppressLint("MissingPermission")
public class GlassBleService extends Service {

    private static final String TAG = "GlassBleService";
    private static final String CHANNEL_ID = "glass_ble_service";
    private static final int NOTIFICATION_ID = 1;

    // ========== CY01 真实 BLE UUID ==========
    private static final UUID UUID_NUS_SERVICE = UUID.fromString("6e40fff0-b5a3-f393-e0a9-e50e24dcca9e");
    private static final UUID UUID_NUS_NOTIFY = UUID.fromString("6e400003-b5a3-f393-e0a9-e50e24dcca9e");
    private static final UUID UUID_SERIAL_PORT_SERVICE = UUID.fromString("de5bf728-d711-4e47-af26-65e3012a5dc7");
    private static final UUID UUID_SERIAL_NOTIFY = UUID.fromString("de5bf729-d711-4e47-af26-65e3012a5dc7");
    private static final UUID UUID_SERIAL_WRITE = UUID.fromString("de5bf72a-d711-4e47-af26-65e3012a5dc7");
    private static final UUID CLIENT_CHARACTERISTIC_CONFIG = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb");

    // ========== 串口帧动作码 ==========
    private static final int ACTION_SYNC_TIME = 64;
    private static final int ACTION_GLASSES_CONTROL = 65;
    private static final int ACTION_GLASSES_BATTERY = 66;
    private static final int ACTION_DEVICE_INFO = 67;
    private static final int ACTION_DEVICE_HEART_BEAT = 69;
    private static final int ACTION_CAMERA_STATUS = 74;
    private static final int ACTION_DEVICE_WEAR = 70;
    private static final int ACTION_DEVICE_WEAR_SUPPORT = 71;
    private static final int ACTION_DEVICE_DATA_REPORTING = 115;
    private static final int ACTION_PICTURE_THUMBNAILS = 0xFD;

    /** 串口写队列间隔，避免 GATT 同时只允许一个未完成写操作 */
    private static final long SERIAL_WRITE_INTERVAL_MS = 120;

    private static final String[] DEVICE_NAME_KEYWORDS = {
            "CY 01", "CY01", "CY1", "CY-01",
            "XyBLE", "Xy3BLE", "XySmart",
            "CY ", "CY-"
    };

    // 主线程 Handler 消息
    private static final int MSG_CHECK_HEARTBEAT = 1001;
    private static final int MSG_SEND_HEARTBEAT = 1006;
    private static final int MSG_RESTART_SCAN = 1005;
    private static final int MSG_CONNECT_TIMEOUT = 1007;

    /** 收到数据超时（超过则认为断开） */
    private static final long HEARTBEAT_TIMEOUT_NORMAL = 30000;
    /** 自己发送心跳包间隔 */
    private static final long HEARTBEAT_SEND_INTERVAL = 8000;
    /** 发起 connectGatt 后多久没收到系统回调则判为超时 */
    private static final long CONNECT_TIMEOUT_MS = 15000;
    private static final long INITIAL_RECONNECT_DELAY_MS = 3000;
    private static final long MAX_RECONNECT_DELAY_MS = 15000;
    /** 自动同步结束后的冷却期，避免退出导入模式触发的 type=1 上报导致死循环 */
    private static final long AUTO_SYNC_COOLDOWN_MS = 2000;
    /** 单次 BLE 扫描持续时长，超时后自动重启，形成不停扫描的循环 */
    private static final long SCAN_DURATION_MS = 8000;

    private final IBinder mBinder = new LocalBinder();

    public class LocalBinder extends Binder {
        public GlassBleService getService() {
            return GlassBleService.this;
        }
    }

    @Override
    public IBinder onBind(Intent intent) {
        return mBinder;
    }

    // ========== 状态 ==========
    private volatile boolean mConnecting = false;
    private long mReconnectDelay = INITIAL_RECONNECT_DELAY_MS;
    private long lastHeartbeatTime = 0;
    private boolean mDataReceivedLogged = false;
    private boolean mScanFirstDeviceLogged = false;
    private boolean mInitCommandsSent = false;

    // 发现的眼镜设备（地址 -> 设备信息），用于多设备时手动选择连接
    private final LinkedHashMap<String, DeviceInfo> mDiscoveredDevices = new LinkedHashMap<>();

    /** 发现的眼镜设备信息 */
    public static class DeviceInfo {
        public final String name;
        public final String address;
        public int rssi;
        public DeviceInfo(String name, String address, int rssi) {
            this.name = name;
            this.address = address;
            this.rssi = rssi;
        }
    }

    // ========== 原生 BLE ==========
    private BluetoothGatt mBluetoothGatt;
    private BluetoothGattCharacteristic mNusNotifyChar;
    private BluetoothGattCharacteristic mSerialWriteChar;
    private BluetoothGattCharacteristic mSerialNotifyChar;

    // 串口写队列（BLE GATT 同一时刻只允许一个未完成写操作，需串行发送）
    private final LinkedList<byte[]> mSerialWriteQueue = new LinkedList<>();
    private volatile boolean mSerialWriting = false;

    // WiFi Direct (P2P) 连接（CY01 照片同步走 P2P + HTTP，而非普通 WiFi 热点）
    private WifiP2pManager mWifiP2pManager;
    private WifiP2pManager.Channel mWifiP2pChannel;
    private BroadcastReceiver mWifiP2pReceiver;
    private boolean mWifiP2pReceiverRegistered = false;
    private volatile boolean mP2pConnecting = false;

    // 眼镜上报的照片列表（用于选择性导入）
    private final List<String> mPhotoList = new ArrayList<>();

    // ===== 拍照分流模式：决定同步完成后照片交给哪个识别链路 =====
    /** 手动同步（UI 按钮）：弹照片勾选框，由用户选择导入 */
    public static final int CAPTURE_MODE_MANUAL = 0;
    /** 普通拍照（语音“拍照”/眼镜按键）：自动导入原图库，不做任何识别 */
    public static final int CAPTURE_MODE_PLAIN = 1;
    /** 二维码拍照（语音“二维码拍照”）：导入后识别最新照片二维码 */
    public static final int CAPTURE_MODE_QR = 2;
    /** 对齐拍照（语音“对齐拍照”）：导入后对最新照片做 YOLO 检测 */
    public static final int CAPTURE_MODE_YOLO = 3;
    /** 万用表拍照（语音“万用表拍照”）：导入后对最新照片做读数识别 */
    public static final int CAPTURE_MODE_METER = 4;

    private volatile int mCaptureMode = CAPTURE_MODE_MANUAL;
    private long mAutoSyncCooldownUntil = 0;

    // 拍照自动识别 mAutoSyncRunnable 声明于 mMainHandler 之后

    // 周期重扫 P2P（眼镜可能在首次扫描后才开启 P2P，这里持续扫描直到连上）
    // 注意：此处不能用 lambda，因为 run() 内需要 postDelayed 调度自身，匿名类才能引用 this
    private final Runnable mRediscoverP2pRunnable = new Runnable() {
        @Override
        public void run() {
            if (AppState.getInstance().isSocketConnected) return;
            if (mWifiP2pManager != null && mWifiP2pChannel != null) {
                try {
                    mWifiP2pManager.discoverPeers(mWifiP2pChannel, null);
                } catch (Exception ignored) {}
            }
            mMainHandler.postDelayed(this, 5000);
        }
    };

    // 原生 BLE 扫描器（SDK 的 startScan 在华为鸿蒙上失效，改用系统原生扫描）
    private BluetoothLeScanner mNativeScanner;
    private ScanCallback mNativeScanCallback;
    private boolean mNativeScanning = false;

    /** 当前 BLE 连接的设备（用于连接后触发经典蓝牙配对，让眼镜音频通道 A2DP/SCO 可用） */
    private BluetoothDevice mBleDevice;

    private final Handler mMainHandler = new Handler(Looper.getMainLooper()) {
        @Override
        public void handleMessage(@NonNull Message msg) {
            switch (msg.what) {
                case MSG_CHECK_HEARTBEAT:
                    handleHeartbeatCheck();
                    mMainHandler.sendEmptyMessageDelayed(MSG_CHECK_HEARTBEAT, 5000);
                    break;
                case MSG_SEND_HEARTBEAT:
                    if (AppState.getInstance().isBleConnected) {
                        writeSerial(ACTION_DEVICE_HEART_BEAT, new byte[]{4, 1});
                        writeSerial(ACTION_GLASSES_BATTERY, new byte[]{0, 0}); // 周期查询电量，实时刷新
                    }
                    mMainHandler.sendEmptyMessageDelayed(MSG_SEND_HEARTBEAT, HEARTBEAT_SEND_INTERVAL);
                    break;
                case MSG_RESTART_SCAN:
                    if (!AppState.getInstance().isBleConnected && !mConnecting) {
                        Log.d(TAG, "Restarting BLE scan...");
                        boolean scanOk = startNativeScan();
                        // 持续扫描：扫描成功则到点自动重启，失败（如蓝牙未开）则短延迟重试，直到连接成功
                        mMainHandler.sendEmptyMessageDelayed(MSG_RESTART_SCAN, scanOk ? SCAN_DURATION_MS : 2000);
                    }
                    break;
                case MSG_CONNECT_TIMEOUT:
                    if (mConnecting && !AppState.getInstance().isBleConnected) {
                        Log.w(TAG, "connectGatt 超时无回调，主动断开重试");
                        postLog("⚠️ 连接超时，重新尝试...");
                        mConnecting = false;
                        if (mBluetoothGatt != null) {
                            try { mBluetoothGatt.close(); } catch (Exception ignored) {}
                            mBluetoothGatt = null;
                        }
                        mMainHandler.sendEmptyMessageDelayed(MSG_RESTART_SCAN, 1000);
                    }
                    break;
            }
        }
    };

    // ===== 照片同步流程状态 =====
    /** 同步流程进行中标志（防止拍照事件与兜底同时触发重复同步） */
    private volatile boolean mSyncActive = false;
    /** 本轮同步开始时刻（用于卡死判定） */
    private volatile long mSyncStartMs = 0;

    /** 收到拍照事件（type=1）后 800ms 自动触发同步（按当前拍照分流模式导入/识别） */
    private final Runnable mAutoSyncRunnable = () -> {
        if (!AppState.getInstance().isBleConnected) {
            postLog("⚠️ 蓝牙未连接，跳过自动同步");
            return;
        }
        postLog("🔄 自动同步最新照片...");
        startPhotoSync();
    };

    private final Runnable mSyncTimeoutRunnable = () -> {
        if (mSyncActive) {
            postLog("⚠️ 照片同步超时（30s，请确认手机 WiFi 已开启）");
            finishSync(0);
        }
    };

    /** WiFi 回传进行中（热点/外部AP）：抑制 P2P 扫描重试，避免框架竞争 */
    private volatile boolean mWifiTransferActive = false;

    // ===== 检测循环/单张检测状态（yolo-fastener 合并恢复） =====
    private volatile boolean mDetectLoopActive = false;
    private volatile boolean mSingleShotActive = false;
    private volatile boolean mDetectNextScheduled = false;
    private static final long DETECT_LOOP_INTERVAL_MS = 2600;
    private final Runnable mDetectFallbackRunnable = new Runnable() {
        @Override
        public void run() {
            // 拍照事件超时未到 → 主动触发一次同步兜底
            if (mDetectLoopActive && !mSyncActive) {
                postLog("⏱️ 拍照事件超时，兜底触发同步");
                startPhotoSync();
            }
        }
    };
    private final Runnable mDetectNextRoundRunnable = () -> {
        if (!mDetectLoopActive) return;
        takePhoto();
    };

    /** P2P 扫描失败重试（reason=0 多为框架忙，退避后重扫） */
    private final Runnable mP2pScanRetryRunnable = () -> {
        if (!mSyncActive) return;
        if (mWifiTransferActive) return; // 热点/外部AP 回传期间不再重试 P2P
        postLog("🔁 重试 P2P 扫描...");
        startWifiP2p();
    };

    private final BluetoothGattCallback mGattCallback = new BluetoothGattCallback() {
        @Override
        public void onConnectionStateChange(BluetoothGatt gatt, int status, int newState) {
            if (newState == BluetoothProfile.STATE_CONNECTED) {
                Log.i(TAG, "GATT connected");
                mConnecting = false;
                mDataReceivedLogged = false;
                mInitCommandsSent = false;
                AppState.getInstance().isBleConnected = true;
                lastHeartbeatTime = System.currentTimeMillis();
                mReconnectDelay = INITIAL_RECONNECT_DELAY_MS;
                mMainHandler.removeMessages(MSG_CONNECT_TIMEOUT);
                mMainHandler.removeMessages(MSG_RESTART_SCAN);
                stopNativeScan();
                updateNotification("已连接: " + AppState.getInstance().bleName);
                EventBus.getDefault().post(new EventMsg(EventMsg.MSG_CONNECT_STATE, 1));
                postLog("✅ BLE已连接，稍后开始发现服务...");
                // 连接成功后自动触发经典蓝牙配对，让眼镜音频通道（A2DP 播报 / SCO 录音）可用
                triggerClassicBond();
                // 参照官方：连接成功后稍作延迟再发现服务，避免过早操作导致连接不稳
                mMainHandler.postDelayed(() -> {
                    if (mBluetoothGatt != null && AppState.getInstance().isBleConnected) {
                        if (!mBluetoothGatt.discoverServices()) {
                            postLog("⚠️ discoverServices 返回 false，重试一次");
                            mMainHandler.postDelayed(() -> {
                                if (mBluetoothGatt != null) {
                                    mBluetoothGatt.discoverServices();
                                }
                            }, 1000);
                        }
                    }
                }, 500);
            } else {
                if (gatt != mBluetoothGatt) return; // 忽略旧设备的断开回调（切换设备时）
                Log.w(TAG, "GATT disconnected status=" + status + " newState=" + newState);
                onGattDisconnected();
            }
        }

        @Override
        public void onServicesDiscovered(BluetoothGatt gatt, int status) {
            if (status != BluetoothGatt.GATT_SUCCESS) {
                postLog("⚠️ 服务发现失败 status=" + status);
                return;
            }

            BluetoothGattService serialService = gatt.getService(UUID_SERIAL_PORT_SERVICE);
            BluetoothGattService nusService = gatt.getService(UUID_NUS_SERVICE);

            if (serialService != null) {
                mSerialWriteChar = serialService.getCharacteristic(UUID_SERIAL_WRITE);
                mSerialNotifyChar = serialService.getCharacteristic(UUID_SERIAL_NOTIFY);
                postLog("✅ 找到串口数据服务 de5bf728");
            } else {
                postLog("⚠️ 未找到串口数据服务 de5bf728");
            }
            if (nusService != null) {
                mNusNotifyChar = nusService.getCharacteristic(UUID_NUS_NOTIFY);
                postLog("✅ 找到NUS控制服务 6e40fff0");
            } else {
                postLog("⚠️ 未找到NUS控制服务 6e40fff0");
            }

            // 订阅通知（官方在服务发现后订阅，不主动请求 MTU，避免个别固件因 MTU 请求断开）
            enableNotification(mSerialNotifyChar);
            enableNotification(mNusNotifyChar);
        }

        @Override
        public void onMtuChanged(BluetoothGatt gatt, int mtu, int status) {
            Log.i(TAG, "MTU changed: " + mtu + " status=" + status);
            postLog("📶 BLE MTU=" + mtu);
        }

        @Override
        public void onDescriptorWrite(BluetoothGatt gatt, BluetoothGattDescriptor descriptor, int status) {
            BluetoothGattCharacteristic ch = descriptor.getCharacteristic();
            if (ch == null) return;
            UUID uuid = ch.getUuid();
            if (UUID_SERIAL_NOTIFY.equals(uuid)) {
                if (status == BluetoothGatt.GATT_SUCCESS) {
                    postLog("✅ 串口数据通道通知已开启");
                    if (!mInitCommandsSent) {
                        mInitCommandsSent = true;
                        mMainHandler.postDelayed(GlassBleService.this::sendInitCommands, 300);
                    }
                } else {
                    postLog("⚠️ 串口通知订阅失败 status=" + status);
                }
            } else if (UUID_NUS_NOTIFY.equals(uuid)) {
                postLog(status == BluetoothGatt.GATT_SUCCESS
                        ? "✅ NUS控制通道通知已开启" : "⚠️ NUS通知订阅失败");
            }
        }

        @Override
        @SuppressWarnings("deprecation")
        public void onCharacteristicChanged(BluetoothGatt gatt, BluetoothGattCharacteristic characteristic) {
            onBleDataReceived(characteristic.getUuid(), characteristic.getValue());
        }

        @Override
        public void onCharacteristicWrite(BluetoothGatt gatt, BluetoothGattCharacteristic characteristic, int status) {
            // 串口写为无响应模式，这里不处理
        }
    };

    // ========== ADB 调试桥（GlassDebugReceiver → 静态入口） ==========

    private static volatile GlassBleService sDebugInstance;

    /** 调试用：暴露当前服务实例（未启动返回 null） */
    public static GlassBleService debugInstance() { return sDebugInstance; }

    /** 调试用：透传原始字节到 BLE 串口写队列（KSDK JSON 帧已是完整帧，直接入队） */
    public static void debugWriteRaw(byte[] data) {
        GlassBleService s = sDebugInstance;
        if (s == null) return;
        synchronized (s.mSerialWriteQueue) {
            s.mSerialWriteQueue.add(data);
        }
        s.processSerialWriteQueue();
    }

    /** 调试用：触发拍照（含先开相机） */
    public static void debugTakePhoto() {
        GlassBleService s = sDebugInstance;
        if (s != null) s.takePhoto();
    }

    /** 调试用：触发照片同步（P2P / 已连接直拉） */
    public static void debugStartSync() {
        GlassBleService s = sDebugInstance;
        if (s != null) s.startPhotoSync();
    }

    /** 调试用：关闭热点 */
    public static void debugStopHotspot() {
        GlassBleService s = sDebugInstance;
        if (s != null) s.stopHotspot();
    }

    /** 调试用：拍照后经 BLE 直传拉取照片（走 YOLO 检测分流） */
    public static void debugBleGet() {
        GlassBleService s = sDebugInstance;
        if (s == null) return;
        s.mCaptureMode = CAPTURE_MODE_YOLO;
        s.takePhoto();
        s.mMainHandler.postDelayed(() -> {
            if (s.mCaptureMode != CAPTURE_MODE_YOLO) return;
            s.postLog("🛠 [BLE直传] 请求推送照片...");
            s.startBleTransfer();
        }, 4000);
    }

    /** 调试用：请求眼镜文件列表（cs_sdfl） */
    public static void debugBleList() {
        GlassBleService s = sDebugInstance;
        if (s != null) {
            s.initBleTransport();
            s.mBleTransport.requestFileList();
        }
    }

    /** 调试用：BLE 文件接收进度 */
    public static String debugBleProgress() {
        GlassBleService s = sDebugInstance;
        return s == null || s.mBleTransport == null ? "服务未就绪" : s.mBleTransport.progress();
    }

    @Override
    public void onCreate() {
        super.onCreate();
        Log.d(TAG, "GlassBleService onCreate");
        sDebugInstance = this;

        createNotificationChannel();
        startForeground(NOTIFICATION_ID, buildNotification("正在搜索AR眼镜..."));

        AppState.getInstance().isBleConnected = false;
        AppState.getInstance().isSystemReady = false;

        logScanDiagnostics();
        boolean scanOk = startNativeScan();
        postLog(scanOk ? "🔍 开始扫描BLE设备..." : "⚠️ 扫描启动失败！");
        postLog("💡 发现眼镜后，请点击「选择眼镜」进行连接");
        // 启动持续扫描循环：不管初始扫描成败，都周期性地重扫，直到连接成功
        mMainHandler.sendEmptyMessageDelayed(MSG_RESTART_SCAN, SCAN_DURATION_MS);

        mMainHandler.sendEmptyMessageDelayed(MSG_CHECK_HEARTBEAT, 10000);
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        sDebugInstance = null;
        mMainHandler.removeCallbacksAndMessages(null);
        stopNativeScan();
        cleanupWifiP2p();
        if (mBluetoothGatt != null) {
            try { mBluetoothGatt.disconnect(); } catch (Exception ignored) {}
            try { mBluetoothGatt.close(); } catch (Exception ignored) {}
            mBluetoothGatt = null;
        }
        mConnecting = false;
        ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE);
    }

    // ========== 通知栏 ==========

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                    CHANNEL_ID, "AR眼镜BLE服务", NotificationManager.IMPORTANCE_LOW);
            channel.setDescription("保持与AR眼镜的BLE连接");
            channel.setShowBadge(false);
            NotificationManager nm = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
            if (nm != null) nm.createNotificationChannel(channel);
        }
    }

    private Notification buildNotification(String text) {
        Intent intent = new Intent(this, MainActivity.class);
        intent.setFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP);
        PendingIntent pi = PendingIntent.getActivity(this, 0, intent,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        return new NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentTitle("AR眼镜连接")
                .setContentText(text)
                .setSmallIcon(android.R.drawable.stat_sys_data_bluetooth)
                .setContentIntent(pi)
                .setOngoing(true)
                .build();
    }

    private void updateNotification(String text) {
        NotificationManager nm = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
        if (nm != null) nm.notify(NOTIFICATION_ID, buildNotification(text));
    }

    // ========== 扫描 ==========

    private void logScanDiagnostics() {
        try {
            BluetoothManager bm = (BluetoothManager) getSystemService(Context.BLUETOOTH_SERVICE);
            BluetoothAdapter adapter = bm != null ? bm.getAdapter() : null;
            boolean btOn = adapter != null && adapter.isEnabled();
            postLog("📱 手机蓝牙: " + (btOn ? "已开启" : "未开启"));

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                boolean scanPerm = ContextCompat.checkSelfPermission(this,
                        android.Manifest.permission.BLUETOOTH_SCAN) == PackageManager.PERMISSION_GRANTED;
                boolean connectPerm = ContextCompat.checkSelfPermission(this,
                        android.Manifest.permission.BLUETOOTH_CONNECT) == PackageManager.PERMISSION_GRANTED;
                postLog("🔑 蓝牙扫描权限(BLUETOOTH_SCAN): " + (scanPerm ? "已授予" : "未授予"));
                postLog("🔑 蓝牙连接权限(BLUETOOTH_CONNECT): " + (connectPerm ? "已授予" : "未授予"));
            } else {
                boolean locPerm = ContextCompat.checkSelfPermission(this,
                        android.Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED;
                postLog("🔑 定位权限(ACCESS_FINE_LOCATION): " + (locPerm ? "已授予" : "未授予"));
            }
        } catch (Exception e) {
            Log.e(TAG, "logScanDiagnostics error", e);
        }
    }

    private void stopNativeScan() {
        try {
            if (mNativeScanner != null && mNativeScanCallback != null && mNativeScanning) {
                mNativeScanner.stopScan(mNativeScanCallback);
            }
        } catch (Exception e) {
            Log.e(TAG, "stopNativeScan error", e);
        }
        mNativeScanning = false;
    }

    private boolean startNativeScan() {
        try {
            BluetoothManager bm = (BluetoothManager) getSystemService(Context.BLUETOOTH_SERVICE);
            BluetoothAdapter adapter = bm != null ? bm.getAdapter() : null;
            if (adapter == null || !adapter.isEnabled()) {
                postLog("⚠️ 原生扫描失败：蓝牙未开启");
                return false;
            }
            mNativeScanner = adapter.getBluetoothLeScanner();
            if (mNativeScanner == null) {
                postLog("⚠️ 原生扫描失败：BluetoothLeScanner不可用");
                return false;
            }

            if (mNativeScanCallback == null) {
                mNativeScanCallback = new ScanCallback() {
                    @Override
                    public void onScanResult(int callbackType, ScanResult result) {
                        handleNativeScanResult(result);
                    }

                    @Override
                    public void onBatchScanResults(List<ScanResult> results) {
                        for (ScanResult r : results) {
                            handleNativeScanResult(r);
                        }
                    }

                    @Override
                    public void onScanFailed(int errorCode) {
                        Log.e(TAG, "Native scan failed: " + errorCode);
                        mNativeScanning = false;
                        if (errorCode == ScanCallback.SCAN_FAILED_ALREADY_STARTED) {
                            return; // 已在扫描中，忽略
                        }
                        postLog("⚠️ 原生扫描失败: errorCode=" + errorCode);
                        // 扫描失败后持续重试，不停止扫描
                        mMainHandler.sendEmptyMessageDelayed(MSG_RESTART_SCAN, 2000);
                    }
                };
            }

            stopNativeScan();
            ScanSettings settings = new ScanSettings.Builder()
                    .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY)
                    .build();
            mNativeScanner.startScan(null, settings, mNativeScanCallback);
            mNativeScanning = true;
            return true;
        } catch (Exception e) {
            Log.e(TAG, "startNativeScan error", e);
            postLog("⚠️ 原生扫描启动失败: " + e.getMessage());
            return false;
        }
    }

    private void handleNativeScanResult(ScanResult result) {
        if (result == null) return;
        BluetoothDevice device = result.getDevice();
        if (device == null) return;
        String deviceName = device.getName();
        short rssi = (short) result.getRssi();

        if (!mScanFirstDeviceLogged) {
            mScanFirstDeviceLogged = true;
            postLog("🔍 扫描已工作，附近有BLE设备: " +
                    (deviceName != null ? deviceName : "(无名设备)") + " rssi=" + rssi);
        }

        if (deviceName != null && !deviceName.isEmpty() && isTargetDevice(deviceName)) {
            addDiscoveredDevice(deviceName, device.getAddress(), rssi);
        }
    }

    /** 记录（去重）发现的眼镜设备，并通知 UI 刷新列表 */
    private void addDiscoveredDevice(String deviceName, String address, int rssi) {
        if (address == null || address.isEmpty()) return;
        boolean isNew = false;
        synchronized (mDiscoveredDevices) {
            DeviceInfo existing = mDiscoveredDevices.get(address);
            if (existing == null) {
                mDiscoveredDevices.put(address, new DeviceInfo(deviceName, address, rssi));
                isNew = true;
            } else {
                existing.rssi = rssi;
            }
        }
        if (isNew) {
            Log.i(TAG, "Found TARGET BLE (native): " + deviceName + " rssi=" + rssi);
            postLog("🎯 发现眼镜: " + deviceName + " (" + address + ")");
        }
    }

    /** 获取当前已发现的眼镜设备列表（快照） */
    public List<DeviceInfo> getDiscoveredDevices() {
        synchronized (mDiscoveredDevices) {
            return new ArrayList<>(mDiscoveredDevices.values());
        }
    }

    /** 获取系统历史配对过的眼镜列表（已配对，含当前未开机的设备） */
    public List<DeviceInfo> getPairedDevices() {
        List<DeviceInfo> result = new ArrayList<>();
        try {
            BluetoothManager bm = (BluetoothManager) getSystemService(Context.BLUETOOTH_SERVICE);
            BluetoothAdapter adapter = bm != null ? bm.getAdapter() : null;
            if (adapter == null || !adapter.isEnabled()) {
                return result;
            }
            for (BluetoothDevice device : adapter.getBondedDevices()) {
                String name = device.getName();
                String addr = device.getAddress();
                if (isTargetDevice(name)) {
                    result.add(new DeviceInfo(name, addr, -127));
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "getPairedDevices error", e);
        }
        return result;
    }

    /** 获取当前已连接的眼镜信息（未连接返回 null） */
    public DeviceInfo getConnectedDevice() {
        if (!AppState.getInstance().isBleConnected) return null;
        String name = AppState.getInstance().bleName;
        String addr = AppState.getInstance().bleAddress;
        return new DeviceInfo(name != null ? name : "", addr != null ? addr : "", 0);
    }

    private boolean isTargetDevice(String deviceName) {
        if (deviceName == null) return false;
        String upperName = deviceName.toUpperCase();
        for (String keyword : DEVICE_NAME_KEYWORDS) {
            if (upperName.contains(keyword.toUpperCase())) return true;
        }
        return false;
    }

    // ========== 连接 ==========

    /** 连接指定眼镜（多设备时由用户手动选择后调用） */
    public void connectToDevice(String bleAddress, String deviceName) {
        if (bleAddress == null || bleAddress.isEmpty()) return;
        if (mConnecting) return; // 正在连接中
        // 已连接其它设备时，先断开旧连接再连新设备
        if (mBluetoothGatt != null) {
            postLog("ℹ️ 切换设备，断开当前连接");
            try { mBluetoothGatt.disconnect(); } catch (Exception ignored) {}
            try { mBluetoothGatt.close(); } catch (Exception ignored) {}
            mBluetoothGatt = null;
        }
        AppState.getInstance().isBleConnected = false;
        mConnecting = true;
        AppState.getInstance().bleName = deviceName;
        AppState.getInstance().bleAddress = bleAddress;
        Log.i(TAG, ">>> Connecting: " + deviceName + " (" + bleAddress + ")");
        updateNotification("正在连接: " + deviceName);
        stopNativeScan();
        try {
            BluetoothManager bm = (BluetoothManager) getSystemService(Context.BLUETOOTH_SERVICE);
            BluetoothAdapter adapter = bm != null ? bm.getAdapter() : null;
            if (adapter == null) {
                mConnecting = false;
                return;
            }
            BluetoothDevice device = adapter.getRemoteDevice(bleAddress);
            mBleDevice = device;
            // 官方用 transport=TRANSPORT_LE 明确走 BLE，避免双模设备 TRANSPORT_AUTO 导致连接不稳
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                mBluetoothGatt = device.connectGatt(this, false, mGattCallback, BluetoothDevice.TRANSPORT_LE);
            } else {
                mBluetoothGatt = device.connectGatt(this, false, mGattCallback);
            }
            if (mBluetoothGatt == null) {
                postLog("⚠️ connectGatt 返回 null");
                mConnecting = false;
                mMainHandler.sendEmptyMessageDelayed(MSG_RESTART_SCAN, 3000);
                return;
            }
            postLog("🔗 发起BLE连接...");
            mMainHandler.removeMessages(MSG_CONNECT_TIMEOUT);
            mMainHandler.sendEmptyMessageDelayed(MSG_CONNECT_TIMEOUT, CONNECT_TIMEOUT_MS);
        } catch (Exception e) {
            Log.e(TAG, "connectGatt error", e);
            postLog("⚠️ 连接失败: " + e.getMessage());
            mConnecting = false;
            mMainHandler.sendEmptyMessageDelayed(MSG_RESTART_SCAN, 3000);
        }
    }

    /**
     * 触发经典蓝牙（BR/EDR）配对。
     * 眼镜是双模设备：BLE 用于数据，经典蓝牙用于音频（A2DP 播报 / SCO 录音）。
     * BLE 连接不会在系统蓝牙列表显示为已配对，需主动 createBond 弹配对框，
     * 配对成功后音频通道才能路由到眼镜扬声器/麦克风。
     */
    private void triggerClassicBond() {
        BluetoothDevice device = mBleDevice;
        if (device == null) return;
        try {
            if (device.getBondState() == BluetoothDevice.BOND_BONDED) {
                postLog("✅ 经典蓝牙已配对，音频通道可用");
                return;
            }
            // createBond 是 @hide API，用反射调用以兼容不同 SDK
            Method createBond = device.getClass().getMethod("createBond");
            createBond.setAccessible(true);
            Object result = createBond.invoke(device);
            String name = device.getName() != null ? device.getName() : device.getAddress();
            postLog("🔗 请求经典蓝牙配对: " + name + " (result=" + result + ")");
        } catch (Exception e) {
            Log.e(TAG, "triggerClassicBond error", e);
            postLog("⚠️ 经典蓝牙配对请求失败: " + e.getMessage());
        }
    }

    private void onGattDisconnected() {
        if (!AppState.getInstance().isBleConnected && mBluetoothGatt == null) {
            return; // 已处理过
        }
        AppState.getInstance().isBleConnected = false;
        AppState.getInstance().isSystemReady = false;
        mConnecting = false;
        mInitCommandsSent = false;
        mDataReceivedLogged = false;
        mNusNotifyChar = null;
        mSerialWriteChar = null;
        mSerialNotifyChar = null;
        mBleDevice = null;

        if (mBluetoothGatt != null) {
            try { mBluetoothGatt.close(); } catch (Exception ignored) {}
            mBluetoothGatt = null;
        }

        EventBus.getDefault().post(new EventMsg(EventMsg.MSG_CONNECT_STATE, 0));
        updateNotification("正在搜索AR眼镜...");

        mMainHandler.removeMessages(MSG_SEND_HEARTBEAT);
        mMainHandler.removeMessages(MSG_RESTART_SCAN);
        mMainHandler.removeMessages(MSG_CONNECT_TIMEOUT);
        mMainHandler.sendEmptyMessageDelayed(MSG_RESTART_SCAN, mReconnectDelay);
        mReconnectDelay = Math.min(mReconnectDelay * 2, MAX_RECONNECT_DELAY_MS);
    }

    private void handleHeartbeatCheck() {
        if (AppState.getInstance().isBleConnected) {
            long timeSince = System.currentTimeMillis() - lastHeartbeatTime;
            if (timeSince > HEARTBEAT_TIMEOUT_NORMAL) {
                Log.w(TAG, "Heartbeat timeout (" + timeSince + "ms), reconnecting...");
                EventBus.getDefault().post(new EventMsg(EventMsg.MSG_TOAST,
                        "蓝牙连接超时，正在重连..."));
                if (mBluetoothGatt != null) {
                    try { mBluetoothGatt.disconnect(); } catch (Exception ignored) {}
                }
            }
        }
    }

    // ========== 协议 ==========

    @SuppressWarnings("deprecation")
    private boolean enableNotification(BluetoothGattCharacteristic ch) {
        if (ch == null || mBluetoothGatt == null) return false;
        try {
            BluetoothGattDescriptor desc = ch.getDescriptor(CLIENT_CHARACTERISTIC_CONFIG);
            if (desc == null) {
                postLog("⚠️ 特征无CCCD: " + ch.getUuid());
                return false;
            }
            if (!mBluetoothGatt.setCharacteristicNotification(ch, true)) {
                postLog("⚠️ setCharacteristicNotification失败: " + ch.getUuid());
                return false;
            }
            desc.setValue(BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE);
            return mBluetoothGatt.writeDescriptor(desc);
        } catch (Exception e) {
            Log.e(TAG, "enableNotification error", e);
            return false;
        }
    }

    private void sendInitCommands() {
        writeSerial(ACTION_GLASSES_BATTERY, new byte[]{0, 0});       // 66 电量
        writeSerial(ACTION_DEVICE_INFO, new byte[]{0, 0});           // 67 设备信息
        writeSerial(ACTION_DEVICE_WEAR_SUPPORT, new byte[]{1, 0});   // 71 穿戴支持
        writeSerial(ACTION_DEVICE_HEART_BEAT, new byte[]{4, 1});     // 69 心跳
        postLog("📤 已发送初始化命令（电量/设备信息/穿戴支持/心跳）");

        // 启动周期心跳
        mMainHandler.removeMessages(MSG_SEND_HEARTBEAT);
        mMainHandler.sendEmptyMessageDelayed(MSG_SEND_HEARTBEAT, HEARTBEAT_SEND_INTERVAL);
    }

    /** 眼镜控制命令（action=65），官方用于开关照片同步 / 重置 P2P 等 */
    private void glassesControl(byte[] payload) {
        writeSerial(ACTION_GLASSES_CONTROL, payload);
    }

    /** 将帧加入串口写队列并触发发送 */
    private void writeSerial(int action, byte[] payload) {
        if (mSerialWriteChar == null || mBluetoothGatt == null) {
            postLog("⚠️ 串口写特征未就绪");
            return;
        }
        byte[] frame = buildSerialPacket(action, payload);
        synchronized (mSerialWriteQueue) {
            mSerialWriteQueue.add(frame);
        }
        processSerialWriteQueue();
    }

    /**
     * 串行发送队列中的帧。BLE GATT 同一时刻只允许一个未完成写操作，
     * 连续调用 writeCharacteristic 会导致后续返回 false（即“写入串口失败”）。
     * 这里每次只发一帧，间隔 SERIAL_WRITE_INTERVAL_MS 后再发下一帧。
     */
    @SuppressWarnings("deprecation")
    private void processSerialWriteQueue() {
        if (mSerialWriting) return;
        byte[] frame;
        synchronized (mSerialWriteQueue) {
            if (mSerialWriteQueue.isEmpty()) return;
            frame = mSerialWriteQueue.poll();
        }
        if (frame == null) return;
        if (mSerialWriteChar == null || mBluetoothGatt == null) {
            mSerialWriting = false;
            return;
        }
        mSerialWriting = true;
        final int action = frame.length > 1 ? (frame[1] & 0xFF) : -1;
        try {
            mSerialWriteChar.setValue(frame);
            mSerialWriteChar.setWriteType(BluetoothGattCharacteristic.WRITE_TYPE_NO_RESPONSE);
            if (!mBluetoothGatt.writeCharacteristic(mSerialWriteChar)) {
                postLog("⚠️ 写入串口失败 action=" + action);
            }
        } catch (Exception e) {
            Log.e(TAG, "writeSerial error", e);
        }
        mMainHandler.postDelayed(() -> {
            mSerialWriting = false;
            processSerialWriteQueue();
        }, SERIAL_WRITE_INTERVAL_MS);
    }

    /** 构造串口 0xBC 帧：[0xBC][action][len低][len高][CRC16低][CRC16高][payload] */
    private byte[] buildSerialPacket(int action, byte[] payload) {
        int len = (payload == null) ? 0 : payload.length;
        byte[] frame = new byte[len + 6];
        frame[0] = (byte) 0xBC;
        frame[1] = (byte) action;
        if (len > 0) {
            frame[2] = (byte) (len & 0xFF);
            frame[3] = (byte) ((len >> 8) & 0xFF);
            int crc = crc16(payload);
            frame[4] = (byte) (crc & 0xFF);
            frame[5] = (byte) ((crc >> 8) & 0xFF);
            System.arraycopy(payload, 0, frame, 6, len);
        } else {
            frame[2] = 0;
            frame[3] = 0;
            frame[4] = (byte) 0xFF;
            frame[5] = (byte) 0xFF;
        }
        return frame;
    }

    /** CRC16/MODBUS，初始值 0xFFFF，反转多项式 0xA001 */
    private int crc16(byte[] data) {
        if (data == null || data.length == 0) return 0xFFFF;
        int crc = 0xFFFF;
        for (byte b : data) {
            crc ^= (b & 0xFF);
            for (int i = 0; i < 8; i++) {
                if ((crc & 1) != 0) {
                    crc = (crc >> 1) ^ 0xA001;
                } else {
                    crc >>= 1;
                }
            }
        }
        return crc & 0xFFFF;
    }

    // ========== 接收 ==========

    private void onBleDataReceived(UUID uuid, byte[] data) {
        lastHeartbeatTime = System.currentTimeMillis();
        if (data == null || data.length == 0) return;

        if (!mDataReceivedLogged) {
            mDataReceivedLogged = true;
            postLog("📡 收到眼镜BLE数据（" + data.length + "字节），通道正常");
            if (!AppState.getInstance().isSystemReady) {
                AppState.getInstance().isSystemReady = true;
                postLog("✅ 数据通道已打通，系统就绪");
                EventBus.getDefault().post(new EventMsg(EventMsg.MSG_SYSTEM_READY));
            }
        }

        if (UUID_SERIAL_NOTIFY.equals(uuid)) {
            logSerialFrame(data);
            // 喂给 BLE 文件协议解析（XyCmd 自动跳过 0xBC 帧垃圾，识别 ##..$$ 文件帧）
            mBleTransport.feed(data);
        } else if (UUID_NUS_NOTIFY.equals(uuid)) {
            postLog("📩 NUS数据: " + toHex(data, 16));
            mBleTransport.feed(data);
        }
    }

    private void logSerialFrame(byte[] data) {
        if (data.length >= 2 && (data[0] & 0xFF) == 0xBC) {
            int action = data[1] & 0xFF;
            int len = 0;
            if (data.length >= 4) {
                len = (data[2] & 0xFF) | ((data[3] & 0xFF) << 8);
            }
            postLog("📩 串口帧 action=" + action + " (" + describeAction(action)
                    + ") 长度=" + len + " 数据=" + toHex(data, 32));
            // 解析眼镜通过 BLE 上报的 IP（数据上报 action=115，首字节 0x08 表示 IP）
            parseDataReporting(action, data);
            // 解析电量上报（action=66）
            parseBattery(action, data);
        } else {
            postLog("📩 串口原始数据: " + toHex(data, 32));
        }
    }

    /** 解析 action=66（电量）响应帧：payload[0]=电量，payload[1]=充电状态 */
    private void parseBattery(int action, byte[] data) {
        if (action != ACTION_GLASSES_BATTERY) return;
        if (data.length < 8) return; // 6 字节帧头 + 至少 2 字节 payload
        int battery = data[6] & 0xFF;
        boolean charging = (data[7] & 0xFF) == 1;
        AppState.getInstance().batteryLevel = battery;
        AppState.getInstance().isCharging = charging;
        postLog("🔋 眼镜电量: " + battery + "%" + (charging ? "（充电中）" : ""));
        EventBus.getDefault().post(new EventMsg(EventMsg.MSG_BATTERY_UPDATE, battery, charging ? 1 : 0));
    }

    /** 解析 action=115（数据上报）帧：type=0x08 时携带眼镜 IP；拍照等其它事件 type 不同 */
    private void parseDataReporting(int action, byte[] data) {
        if (action != ACTION_DEVICE_DATA_REPORTING) return;
        int type = data.length > 6 ? (data[6] & 0xFF) : -1;
        postLog("📡 数据上报 type=" + type + " 完整=" + toHex(data, data.length));
        if (data.length >= 11 && type == 0x08) {
            int a = data[7] & 0xFF;
            int b = data[8] & 0xFF;
            int c = data[9] & 0xFF;
            int d = data[10] & 0xFF;
            String ip = a + "." + b + "." + c + "." + d;
            postLog("🎯 眼镜上报IP: " + ip);
            onGlassesIpObtained(ip);
        } else if (type == 0x01 && data.length >= 8) {
            // 拍照事件：data[7] 为照片序号
            onPhotoCaptured(data[7] & 0xFF);
        }
    }

    private void onGlassesIpObtained(String ip) {
        if (ip == null || ip.isEmpty()) return;
        AppState.getInstance().serverIp = ip;
        AppState.getInstance().isSocketConnected = true;
        postLog("✅ 眼镜IP已获取: " + ip);
        EventBus.getDefault().post(new EventMsg(EventMsg.MSG_WIFI_CONNECT_RESULT, 1));
        fetchPhotoList();
    }

    private String describeAction(int action) {
        switch (action) {
            case ACTION_SYNC_TIME: return "同步时间";
            case ACTION_GLASSES_CONTROL: return "眼镜控制";
            case ACTION_GLASSES_BATTERY: return "电量";
            case ACTION_DEVICE_INFO: return "设备信息";
            case ACTION_DEVICE_HEART_BEAT: return "心跳";
            case ACTION_DEVICE_WEAR: return "佩戴状态";
            case ACTION_DEVICE_WEAR_SUPPORT: return "穿戴支持";
            case ACTION_DEVICE_DATA_REPORTING: return "数据上报";
            case ACTION_PICTURE_THUMBNAILS: return "照片缩略图";
            default: return "未知";
        }
    }

    private String toHex(byte[] data, int maxBytes) {
        if (data == null) return "";
        StringBuilder sb = new StringBuilder();
        int n = Math.min(data.length, maxBytes);
        for (int i = 0; i < n; i++) {
            sb.append(String.format("%02X ", data[i] & 0xFF));
        }
        return sb.toString().trim();
    }

    // ========== 日志 ==========

    private void postLog(String text) {
        Log.i("GlassLog", text); // 同步写入 logcat，便于 adb 远程诊断
        mMainHandler.post(() -> EventBus.getDefault().post(new EventMsg(EventMsg.MSG_LOG, text)));
    }

    private void toast(String text) {
        mMainHandler.post(() -> EventBus.getDefault().post(new EventMsg(EventMsg.MSG_TOAST, text)));
    }

    // ========== 外部 API（供 MainActivity 调用） ==========

    /** 一键同步照片：BLE 通知眼镜开启照片同步（进入照片导入模式）→ 手机通过 WiFi Direct
     * 连接眼镜 → 通过 HTTP 拉取照片列表 → 用户勾选后选择性下载。 */
    public void syncPhotos() {
        if (!AppState.getInstance().isBleConnected) {
            toast("请先等待蓝牙连接眼镜");
            return;
        }
        mCaptureMode = CAPTURE_MODE_MANUAL; // 手动同步：弹照片勾选框
        startPhotoSync();
    }

    /**
     * 带分流模式的拍照：拍照 → 同步到手机原图库 → 按模式识别。
     * @param mode {@link #CAPTURE_MODE_PLAIN} 仅存原图库 / {@link #CAPTURE_MODE_QR} 二维码 /
     *             {@link #CAPTURE_MODE_YOLO} YOLO 检测 / {@link #CAPTURE_MODE_METER} 万用表读数
     */
    public void takePhotoFor(int mode) {
        if (!AppState.getInstance().isBleConnected) {
            toast("请先等待蓝牙连接眼镜");
            return;
        }
        if (mSyncActive) {
            toast("上一张还在同步中，请稍候");
            return;
        }
        mCaptureMode = mode;
        postLog("📸 拍照（模式=" + modeName(mode) + "）...");
        takePhoto();
        // 4 秒后走「同步到手机」管线兜底（部分固件不回报拍照事件）
        mMainHandler.postDelayed(() -> {
            if (mSyncActive || mCaptureMode != mode) return;
            postLog("📷 未收到拍照事件，主动同步照片");
            startPhotoSync();
        }, 4000);
    }

    public static String modeName(int mode) {
        switch (mode) {
            case CAPTURE_MODE_QR: return "二维码识别";
            case CAPTURE_MODE_YOLO: return "YOLO检测";
            case CAPTURE_MODE_METER: return "万用表读数";
            case CAPTURE_MODE_PLAIN: return "存原图库";
            default: return "手动同步";
        }
    }

    /** 拍照命令：官方“耳机模式”下需先确保相机开启（action=74 {2,1,1}），再发拍照（action=65 {2,1,1}） */
    public void takePhoto() {
        if (!AppState.getInstance().isBleConnected) {
            toast("请先等待蓝牙连接眼镜");
            return;
        }
        postLog("📸 拍照：先开启相机...");
        // 1. 开启相机（earphoneCameraStatusSetting(true,false) => action=74, payload={2,1,1}）
        writeSerial(ACTION_CAMERA_STATUS, new byte[]{2, 1, 1});
        // 2. 延时后发送拍照命令，给相机开启留出时间
        mMainHandler.removeCallbacks(mTakePhotoRunnable);
        mMainHandler.postDelayed(mTakePhotoRunnable, 600);
    }

    private final Runnable mTakePhotoRunnable = () -> {
        postLog("📸 发送拍照命令...");
        glassesControl(new byte[]{2, 1, 1});
    };

    /**
     * 核心流程（CY01 照片同步走 WiFi Direct + HTTP，而非普通 WiFi 热点）：
     * 1. BLE 发送 glassesControl({2,1,4,1}) 让眼镜进入照片导入模式（开启 P2P 组网）
     * 2. 启动 WiFi Direct（P2P）扫描，发现并连接眼镜
     * 3. 连接成功后从 WifiP2pInfo.groupOwnerAddress 获取眼镜 IP
     * 4. 拉取照片列表，交由用户选择后下载
     */
    private void startPhotoSync() {
        // 防重入：同步流程进行中忽略重复触发（拍照事件与检测循环兜底可能同时到达）
        if (mSyncActive) {
            postLog("ℹ️ 同步进行中，忽略重复触发");
            return;
        }
        mSyncActive = true;
        mSyncStartMs = System.currentTimeMillis();
        if (AppState.getInstance().serverIp != null
                && !AppState.getInstance().serverIp.isEmpty()
                && AppState.getInstance().isSocketConnected) {
            fetchPhotoList();
            return;
        }

        mP2pConnecting = false;
        postLog("🔄 正在开启眼镜照片同步（WiFi Direct）...");
        // 官方导入照片命令：glassesControl(new byte[]{2, 1, 4, 1})
        // 注意：此处不要 removeGroup——它会触发 P2P 框架重置，随后的扫描会 reason=0 失败；
        // 残留组的清理放在连接超时路径（mP2pConnectTimeoutRunnable）处理
        glassesControl(new byte[]{2, 1, 4, 1});

        // 等眼镜开启 P2P 后再扫描
        mMainHandler.postDelayed(this::startWifiP2p, 1500);
    }

    /** 计算眼镜 WiFi/P2P 名称（设备名_MAC），用于 P2P 设备匹配 */
    private String computeGlassesWifiSsid() {
        AppState st = AppState.getInstance();
        String name = st.bleName != null ? st.bleName : "";
        String mac = st.bleAddress != null ? st.bleAddress.replace(":", "") : "";
        String ssid;
        if (name.contains("_")) {
            String[] parts = name.split("_");
            String str = parts.length > 2 ? parts[parts.length - 1] : parts[0];
            if (str.length() > 20) str = str.substring(0, 20);
            ssid = str + "_" + mac;
        } else {
            ssid = name + "_" + mac;
        }
        return ssid;
    }

    /** 初始化并启动 WiFi Direct 扫描 */
    private void startWifiP2p() {
        try {
            // WiFi Direct 依赖手机 WiFi 开启（P2P 扫描失败 reason=0 的最常见原因）
            WifiManager wm = (WifiManager) getApplicationContext().getSystemService(Context.WIFI_SERVICE);
            if (wm != null && !wm.isWifiEnabled()) {
                postLog("⚠️ 手机 WiFi 未开启！WiFi Direct 需要 WiFi 打开才能工作，请下拉通知栏开启 WiFi");
            }
            mWifiP2pManager = (WifiP2pManager) getSystemService(Context.WIFI_P2P_SERVICE);
            if (mWifiP2pManager == null) {
                postLog("⚠️ 设备不支持 WiFi Direct");
                EventBus.getDefault().post(new EventMsg(EventMsg.MSG_WIFI_CONNECT_RESULT, 0));
                return;
            }
            if (mWifiP2pChannel == null) {
                mWifiP2pChannel = mWifiP2pManager.initialize(this, Looper.getMainLooper(), null);
            }
            registerWifiP2pReceiver();
            postLog("🔍 开始扫描眼镜 P2P 设备...");
            mWifiP2pManager.discoverPeers(mWifiP2pChannel, new WifiP2pManager.ActionListener() {
                @Override
                public void onSuccess() {
                    postLog("ℹ️ P2P 扫描已启动");
                }

                @Override
                public void onFailure(int reason) {
                    // reason=0 多为 P2P 框架忙/未就绪，3 秒后自动重试直到同步超时
                    if (mWifiTransferActive) {
                        // 热点/外部AP 回传期间跳过 P2P 重试，避免 WiFi 框架竞争（日志刷屏 reason=2 的根源）
                        return;
                    }
                    postLog("⚠️ P2P 扫描失败: " + reason + "，3 秒后自动重试...");
                    mMainHandler.removeCallbacks(mP2pScanRetryRunnable);
                    mMainHandler.postDelayed(mP2pScanRetryRunnable, 3000);
                }
            });
        } catch (Exception e) {
            Log.e(TAG, "startWifiP2p error", e);
            postLog("⚠️ 启动 P2P 失败: " + e.getMessage());
            EventBus.getDefault().post(new EventMsg(EventMsg.MSG_WIFI_CONNECT_RESULT, 0));
        }
    }

    private void registerWifiP2pReceiver() {
        if (mWifiP2pReceiverRegistered) return;
        IntentFilter filter = new IntentFilter();
        filter.addAction(WifiP2pManager.WIFI_P2P_STATE_CHANGED_ACTION);
        filter.addAction(WifiP2pManager.WIFI_P2P_PEERS_CHANGED_ACTION);
        filter.addAction(WifiP2pManager.WIFI_P2P_CONNECTION_CHANGED_ACTION);
        mWifiP2pReceiver = new BroadcastReceiver() {
            @Override
            @SuppressWarnings("deprecation")
            public void onReceive(Context context, Intent intent) {
                String action = intent == null ? null : intent.getAction();
                if (action == null) return;
                if (WifiP2pManager.WIFI_P2P_STATE_CHANGED_ACTION.equals(action)) {
                    int state = intent.getIntExtra(WifiP2pManager.EXTRA_WIFI_STATE, -1);
                    postLog("📡 P2P 可用性: " + (state == WifiP2pManager.WIFI_P2P_STATE_ENABLED
                            ? "可用" : "不可用 state=" + state));
                } else if (WifiP2pManager.WIFI_P2P_PEERS_CHANGED_ACTION.equals(action)) {
                    onWifiP2pPeersChanged();
                } else if (WifiP2pManager.WIFI_P2P_CONNECTION_CHANGED_ACTION.equals(action)) {
                    android.net.NetworkInfo netInfo = intent.getParcelableExtra(
                            WifiP2pManager.EXTRA_NETWORK_INFO);
                    WifiP2pInfo info = intent.getParcelableExtra(WifiP2pManager.EXTRA_WIFI_P2P_INFO);
                    boolean netConnected = netInfo != null && netInfo.isConnected();
                    postLog("📡 P2P 连接状态变化: net=" + netConnected
                            + " groupFormed=" + (info != null && info.groupFormed)
                            + " owner=" + (info != null && info.groupOwnerAddress != null
                            ? info.groupOwnerAddress.getHostAddress() : "null"));
                    if (info != null && info.groupFormed && netConnected) {
                        onWifiP2pConnected(info);
                    } else if (!netConnected) {
                        mP2pConnecting = false;
                    }
                }
            }
        };
        registerReceiver(mWifiP2pReceiver, filter);
        mWifiP2pReceiverRegistered = true;
    }

    private void onWifiP2pPeersChanged() {
        if (mWifiP2pManager == null || mWifiP2pChannel == null || mP2pConnecting) return;
        mWifiP2pManager.requestPeers(mWifiP2pChannel, peers -> {
            Collection<WifiP2pDevice> list = peers.getDeviceList();
            for (WifiP2pDevice device : list) {
                if (matchesGlassesP2p(device)) {
                    postLog("✅ 发现眼镜 P2P 设备: " + device.deviceName);
                    connectToP2pDevice(device);
                    return;
                }
            }
        });
    }

    private boolean matchesGlassesP2p(WifiP2pDevice device) {
        String name = device == null ? null : device.deviceName;
        if (name == null) return false;
        String wifiName = computeGlassesWifiSsid();
        if (!wifiName.isEmpty() && name.equalsIgnoreCase(wifiName)) return true;
        String mac = AppState.getInstance().bleAddress != null
                ? AppState.getInstance().bleAddress.replace(":", "") : "";
        if (!mac.isEmpty() && name.endsWith(mac)) return true;
        String bleName = AppState.getInstance().bleName != null
                ? AppState.getInstance().bleName.replace(" ", "") : "";
        if (!bleName.isEmpty() && name.replace(" ", "").startsWith(bleName)) return true;
        return false;
    }

    private void connectToP2pDevice(WifiP2pDevice device) {
        if (mP2pConnecting) return;
        mP2pConnecting = true;
        try {
            WifiP2pConfig config = new WifiP2pConfig();
            config.deviceAddress = device.deviceAddress;
            config.wps.setup = 0; // PBC，与官方一致
            mWifiP2pManager.connect(mWifiP2pChannel, config, new WifiP2pManager.ActionListener() {
                @Override
                public void onSuccess() {
                    postLog("📶 P2P 连接请求已发送");
                    // 连接超时兜底：残留组导致的静默无响应，清除组后重试扫描
                    mMainHandler.removeCallbacks(mP2pConnectTimeoutRunnable);
                    mMainHandler.postDelayed(mP2pConnectTimeoutRunnable, P2P_CONNECT_TIMEOUT_MS);
                }

                @Override
                public void onFailure(int reason) {
                    mP2pConnecting = false;
                    postLog("⚠️ P2P 连接失败: " + reason);
                }
            });
        } catch (Exception e) {
            mP2pConnecting = false;
            postLog("⚠️ P2P 连接异常: " + e.getMessage());
        }
    }

    /** P2P 连接超时：请求发出后一直无回调（多为残留组/眼镜端未响应），清组重试 */
    private static final long P2P_CONNECT_TIMEOUT_MS = 12000;
    private final Runnable mP2pConnectTimeoutRunnable = new Runnable() {
        @Override
        public void run() {
            if (!mSyncActive) return;
            postLog("⚠️ P2P 连接 12s 无响应，清除残留组并重试...");
            if (mWifiP2pManager != null && mWifiP2pChannel != null) {
                try { mWifiP2pManager.removeGroup(mWifiP2pChannel, null); } catch (Exception ignored) {}
                try { mWifiP2pManager.cancelConnect(mWifiP2pChannel, null); } catch (Exception ignored) {}
                try { mWifiP2pManager.discoverPeers(mWifiP2pChannel, null); } catch (Exception ignored) {}
            }
            mP2pConnecting = false;
        }
    };

    private void onWifiP2pConnected(WifiP2pInfo info) {
        mP2pConnecting = false;
        mMainHandler.removeCallbacks(mP2pConnectTimeoutRunnable);
        if (info == null || !info.groupFormed) {
            postLog("⚠️ P2P 未成功组网");
            return;
        }
        if (info.isGroupOwner) {
            // 手机成为组长时，眼镜作为客户端加入，其 IP 会通过 BLE action=115(type=8) 上报
            postLog("ℹ️ 手机成为 P2P 组长，等待眼镜上报 IP...");
            return;
        }
        // 眼镜为组长时，groupOwnerAddress 即眼镜 IP（兜底）
        if (info.groupOwnerAddress != null) {
            onGlassesIpObtained(info.groupOwnerAddress.getHostAddress());
        }
    }

    private void cleanupWifiP2p() {
        mMainHandler.removeCallbacks(mRediscoverP2pRunnable);
        mMainHandler.removeCallbacks(mP2pScanRetryRunnable);
        try {
            if (mWifiP2pReceiverRegistered && mWifiP2pReceiver != null) {
                unregisterReceiver(mWifiP2pReceiver);
                mWifiP2pReceiverRegistered = false;
                mWifiP2pReceiver = null;
            }
            if (mWifiP2pManager != null && mWifiP2pChannel != null) {
                mWifiP2pManager.removeGroup(mWifiP2pChannel, null);
            }
            if (mWifiP2pChannel != null) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
                    mWifiP2pChannel.close();
                }
                mWifiP2pChannel = null;
            }
        } catch (Exception e) {
            Log.w(TAG, "cleanupWifiP2p error", e);
        }
        mWifiP2pManager = null;
        mP2pConnecting = false;
    }

    /** 通过 HTTP 拉取眼镜照片列表（media.config，失败回退 vf_list.txt），并通知 UI 展示勾选 */
    private void fetchPhotoList() {
        final String ip = AppState.getInstance().serverIp;
        if (ip == null || ip.isEmpty()) {
            postLog("⚠️ 眼镜IP未知，无法获取照片列表");
            finishSync(0);
            return;
        }
        new Thread(() -> {
            try {
                // 参照官方：拿到 IP 后稍作延迟，等眼镜 HTTP 服务就绪
                Thread.sleep(1000);

                String[] files = null;
                String fileBaseUrl = "";
                String config = null;

                for (int attempt = 1; attempt <= 3 && (config == null || config.trim().isEmpty()); attempt++) {
                    config = httpGet("http://" + ip + "/files/media.config");
                    if (config == null || config.trim().isEmpty()) {
                        postLog("⚠️ media.config 未获取到(第" + attempt + "次)");
                        if (attempt < 3) Thread.sleep(500);
                    }
                }

                if (config != null && !config.trim().isEmpty()) {
                    files = splitLines(config);
                    fileBaseUrl = "http://" + ip + "/files/";
                } else {
                    postLog("ℹ️ media.config 为空，尝试 vf_list.txt...");
                    config = httpGet("http://" + ip + ":80/storage/sd0/C/DCIM/1/vf_list.txt");
                    if (config != null && !config.trim().isEmpty()) {
                        files = splitLines(config);
                        fileBaseUrl = "http://" + ip + ":80/storage/sd0/C/DCIM/1/";
                    }
                }

                if (files == null || files.length == 0) {
                    postLog("⚠️ 未获取到照片列表（眼镜里可能没有照片，或文件服务未就绪）");
                    EventBus.getDefault().post(new EventMsg(EventMsg.MSG_PHOTO_LIST, new ArrayList<String>()));
                    finishSync(0);
                    return;
                }

                synchronized (mPhotoList) {
                    mPhotoList.clear();
                    for (String name : files) {
                        String t = name.trim();
                        if (!t.isEmpty()) mPhotoList.add(t);
                    }
                }
                // 先把所有照片下载到临时目录，供勾选弹窗展示缩略图
                downloadAllToTemp(fileBaseUrl);
            } catch (Exception e) {
                Log.e(TAG, "fetchPhotoList error", e);
                postLog("⚠️ 获取照片列表失败: " + e.getMessage());
                finishSync(0);
            }
        }).start();
    }

    /** 下载全部照片到临时目录，供勾选弹窗展示缩略图 */
    private void downloadAllToTemp(String fileBaseUrl) {
        List<String> names;
        synchronized (mPhotoList) {
            names = new ArrayList<>(mPhotoList);
        }
        File tmpDir = getTmpDir();
        clearDir(tmpDir);
        int total = names.size();
        int success = 0;
        for (String name : names) {
            String n = name.trim();
            if (n.isEmpty()) continue;
            File out = new File(tmpDir, n);
            if (httpDownload(fileBaseUrl + n, out, n)) {
                success++;
                postLog("📥 下载缩略图 " + success + "/" + total + ": " + n);
            }
        }
        if (mCaptureMode != CAPTURE_MODE_MANUAL) {
            // 语音拍照分流模式：下载完直接全部导入，不弹勾选框
            postLog("📥 缩略图下载完成 " + success + "/" + total + "，自动导入全部照片");
            finalizeImport(names);
        } else {
            postLog("📥 缩略图下载完成 " + success + "/" + total + "，请选择要导入的照片");
            EventBus.getDefault().post(new EventMsg(EventMsg.MSG_PHOTO_LIST, names));
        }
    }

    /** 清空目录内容 */
    @SuppressWarnings("ResultOfMethodCallIgnored")
    private void clearDir(File dir) {
        if (dir == null || !dir.exists()) return;
        File[] files = dir.listFiles();
        if (files == null) return;
        for (File f : files) {
            if (f.isDirectory()) clearDir(f);
            else f.delete();
        }
    }

    @SuppressWarnings("ResultOfMethodCallIgnored")
    private File getTmpDir() {
        File dir = new File(getExternalFilesDir(null), "glass_media/tmp");
        if (!dir.exists()) dir.mkdirs();
        return dir;
    }

    /** 保留用户勾选的照片（移动到正式目录），删除未勾选的临时文件 */
    @SuppressWarnings("ResultOfMethodCallIgnored")
    public void finalizeImport(List<String> selected) {
        final List<String> sel = (selected == null) ? new ArrayList<>() : selected;
        postLog("📥 保留 " + sel.size() + " 张照片...");
        new Thread(() -> {
            try {
                Set<String> keep = new HashSet<>(sel);
                File tmpDir = getTmpDir();
                File photoDir = getPhotoDir();
                int kept = 0;
                List<String> names;
                synchronized (mPhotoList) {
                    names = new ArrayList<>(mPhotoList);
                }
                for (String name : names) {
                    String n = name.trim();
                    if (n.isEmpty()) continue;
                    File src = new File(tmpDir, n);
                    if (!src.exists()) continue;
                    if (keep.contains(n)) {
                        File dst = new File(photoDir, n);
                        if (moveFile(src, dst)) {
                            kept++;
                            EventBus.getDefault().post(new EventMsg(EventMsg.MSG_FILE_RECV_FINISH, dst.getAbsolutePath()));
                        }
                    } else {
                        src.delete();
                    }
                }
                postLog("📥 照片导入完成: 保留 " + kept + " 张");
                finishSync(kept);
            } catch (Exception e) {
                Log.e(TAG, "finalizeImport error", e);
                postLog("⚠️ 导入失败: " + e.getMessage());
                finishSync(0);
            }
        }).start();
    }

    /** 移动文件（跨目录），renameTo 失败时降级为复制 */
    @SuppressWarnings("ResultOfMethodCallIgnored")
    private boolean moveFile(File src, File dst) {
        try {
            if (dst.exists()) dst.delete();
            if (src.renameTo(dst)) return true;
            InputStream in = new java.io.FileInputStream(src);
            FileOutputStream out = new FileOutputStream(dst);
            byte[] buf = new byte[8192];
            int len;
            while ((len = in.read(buf)) > 0) out.write(buf, 0, len);
            out.flush();
            out.close();
            in.close();
            src.delete();
            return dst.exists();
        } catch (Exception e) {
            Log.e(TAG, "moveFile error", e);
            return false;
        }
    }

    /** 用户在勾选弹窗中取消：清理临时文件并断开连接 */
    public void cancelSync() {
        postLog("ℹ️ 已取消本次导入");
        new Thread(() -> {
            clearDir(getTmpDir());
            finishSync(0);
        }).start();
    }

    /** 同步结束（成功或失败）：复位连接状态并清理 P2P，避免 UI 卡在“传输中” */
    private void finishSync(int count) {
        mSyncActive = false;
        mMainHandler.removeCallbacks(mSyncTimeoutRunnable);
        // 热点回传路径收尾：关闭手机热点
        mHotspotTransferActive = false;
        mWifiTransferActive = false;
        EventBus.getDefault().post(new EventMsg(EventMsg.MSG_TRANSFER_PROGRESS, -1, ""));
        stopHotspot();
        AppState.getInstance().isSocketConnected = false;
        mMainHandler.post(this::cleanupWifiP2p);
        // 通知眼镜退出照片导入模式，恢复正常拍照操作（官方 fileDownloadComplete 使用的命令）
        mMainHandler.post(() -> {
            glassesControl(new byte[]{2, 1, 9});
            postLog("ℹ️ 已通知眼镜退出照片导入模式");
        });
        EventBus.getDefault().post(new EventMsg(EventMsg.MSG_SYNC_COMPLETE, count));

        // 拍照分流：同步结束后按模式把最新照片交给对应识别链路
        int mode = mCaptureMode;
        mCaptureMode = CAPTURE_MODE_MANUAL; // 本轮结束即复位，避免影响下一次拍照
        if (mode != CAPTURE_MODE_MANUAL) {
            mAutoSyncCooldownUntil = System.currentTimeMillis() + AUTO_SYNC_COOLDOWN_MS;
        }
        if (count <= 0) {
            if (mode == CAPTURE_MODE_QR || mode == CAPTURE_MODE_YOLO || mode == CAPTURE_MODE_METER) {
                postLog("⚠️ " + modeName(mode) + "：未获取到照片（P2P/网络未就绪）");
                if (mode == CAPTURE_MODE_YOLO) {
                    EventBus.getDefault().post(new EventMsg(EventMsg.MSG_DETECT_RESULT,
                            new com.ar.glass.vision.DetectResult("未获取到照片")));
                }
            }
            return;
        }
        switch (mode) {
            case CAPTURE_MODE_QR:
                recognizeLatestPhoto();
                break;
            case CAPTURE_MODE_YOLO:
                detectLatestPhotoWithYolo();
                break;
            case CAPTURE_MODE_METER:
                dispatchMeterRecognition();
                break;
            case CAPTURE_MODE_PLAIN:
            default:
                postLog("📷 照片已存入原图库（未做识别）");
                break;
        }
    }

    /** 万用表分流：把最新照片路径发给 UI 层做云端读数识别 */
    private void dispatchMeterRecognition() {
        File latest = findLatestPhoto();
        if (latest == null) {
            postLog("⚠️ 万用表识别：未找到照片");
            return;
        }
        postLog("� 万用表读数识别: " + latest.getName());
        EventBus.getDefault().post(new EventMsg(EventMsg.MSG_METER_RECOGNIZE, latest.getAbsolutePath()));
    }

    // ===== BLE 直传（ksdk 文件协议：cs_asfl 请求 → FileMessage 分块 → cs_flts 确认） =====

    private BleFileTransport mBleTransport;
    private volatile boolean mBleTransferActive = false;

    private void initBleTransport() {
        if (mBleTransport != null) return;
        mBleTransport = new BleFileTransport(new BleFileTransport.Callback() {
            @Override
            public void onLog(String msg) {
                postLog(msg);
            }

            @Override
            public void onFileReady(File file) {
                mMainHandler.post(() -> onBleFileReceived(file));
            }

            @Override
            public void onProgress(int percent, String info) {
                EventBus.getDefault().post(new EventMsg(EventMsg.MSG_TRANSFER_PROGRESS, percent, info));
            }
        });
        mBleTransport.setWriter(data -> {
            if (mSerialWriteChar == null || mBluetoothGatt == null) {
                Log.w(TAG, "BLE transport write: 串口未就绪");
                return;
            }
            synchronized (mSerialWriteQueue) {
                mSerialWriteQueue.add(data);
            }
            processSerialWriteQueue();
        });
    }

    /** BLE 直传：请求眼镜把最新照片经 BLE 分块推送（无需任何 WiFi） */
    private void startBleTransfer() {
        initBleTransport();
        mSyncActive = true;
        mSyncStartMs = System.currentTimeMillis();
        mBleTransferActive = true;
        mBleTransport.requestFile(1); // ftype=1 照片（实测校正）
        EventBus.getDefault().post(new EventMsg(EventMsg.MSG_TRANSFER_PROGRESS, -2, "BLE 直传：等待眼镜推送..."));
        mMainHandler.removeCallbacks(mBleTransferTimeoutRunnable);
        mMainHandler.postDelayed(mBleTransferTimeoutRunnable, 60000);
    }

    private final Runnable mBleTransferTimeoutRunnable = () -> {
        if (!mBleTransferActive) return;
        mBleTransferActive = false;
        postLog("⚠️ [BLE直传] 60s 未收到文件，回退热点模式");
        EventBus.getDefault().post(new EventMsg(EventMsg.MSG_TRANSFER_PROGRESS, -2, "BLE 超时，切换热点回传..."));
        startHotspotTransfer();
    };

    /** BLE 文件接收完成：按当前分流模式识别（与 WiFi 链路收尾解耦） */
    private void onBleFileReceived(File file) {
        mBleTransferActive = false;
        mMainHandler.removeCallbacks(mBleTransferTimeoutRunnable);
        EventBus.getDefault().post(new EventMsg(EventMsg.MSG_TRANSFER_PROGRESS, -1, ""));
        int mode = mCaptureMode;
        mCaptureMode = CAPTURE_MODE_MANUAL;
        if (mode == CAPTURE_MODE_MANUAL) return;
        mSyncActive = false;
        postLog("📷 [BLE直传] 收到照片: " + file.getName() + "（" + modeName(mode) + "）");
        switch (mode) {
            case CAPTURE_MODE_QR:
                recognizeLatestPhoto();
                break;
            case CAPTURE_MODE_METER:
                dispatchMeterRecognition();
                break;
            case CAPTURE_MODE_PLAIN:
                postLog("📷 照片已存入原图库（未做识别）");
                break;
            case CAPTURE_MODE_YOLO:
            default:
                detectLatestPhotoWithYolo();
                break;
        }
    }

    // ===== 热点配网回传（LocalOnlyHotspot + BLE 配网 + ARP 发现，绕开 P2P 封锁） =====

    private WifiManager.LocalOnlyHotspotReservation mHotspotReservation;
    private String mHotspotSsid;
    private String mHotspotPass;
    private com.xy.ksdk.api.cmd.CmdSendManager mCmdSender;
    private volatile boolean mHotspotTransferActive = false;
    private int mHotspotScanRound = 0;
    /** 热点模式下轮询网段的轮数（×2s 间隔，120 秒发现窗口） */
    private static final int HOTSPOT_SCAN_ROUNDS = 60;
    /** 热点路径同步总超时（放宽：配网+关联+拉取需要较长时间） */
    private static final long HOTSPOT_SYNC_TIMEOUT_MS = 180000;

    /** ksdk 命令桥接：cs_ JSON 命令复用现有 BLE 串口写队列 */
    private void initCmdSender() {
        if (mCmdSender != null) return;
        try {
            mCmdSender = com.xy.ksdk.api.cmd.CmdSendManager.getInstance();
            mCmdSender.init(getApplicationContext());
            mCmdSender.setBluetooth(new com.xy.ksdk.api.cmd.IBluetooth() {
                @Override
                public void write(byte[] data) {
                    String text = new String(data);
                    Log.i("GlassLog", "📤 [KSDK] 发送: " + text);
                    if (mSerialWriteChar == null || mBluetoothGatt == null) {
                        Log.i("GlassLog", "📤 [KSDK] 串口未就绪，丢弃");
                        return;
                    }
                    synchronized (mSerialWriteQueue) {
                        mSerialWriteQueue.add(data);
                    }
                    processSerialWriteQueue();
                }
            });
            postLog("✅ KSDK 命令桥接就绪");
        } catch (Throwable e) {
            Log.e(TAG, "initCmdSender error", e);
            postLog("⚠️ KSDK 桥接失败: " + e.getMessage());
        }
    }

    /** 热点回传：开热点 → BLE 配网让眼镜 STA 连入 → ARP 发现 → 复用拉取/检测链路 */
    private void startHotspotTransfer() {
        mSyncActive = true;
        mSyncStartMs = System.currentTimeMillis();
        mMainHandler.removeCallbacks(mSyncTimeoutRunnable);
        mMainHandler.postDelayed(mSyncTimeoutRunnable, HOTSPOT_SYNC_TIMEOUT_MS);
        if (Build.VERSION.SDK_INT < 26) {
            postLog("⚠️ 系统低于 8.0 不支持热点模式，回退 P2P");
            startPhotoSync();
            return;
        }
        WifiManager wm = (WifiManager) getApplicationContext().getSystemService(Context.WIFI_SERVICE);
        if (wm == null) {
            postLog("⚠️ 无 WiFi 服务，回退 P2P");
            startPhotoSync();
            return;
        }
        mHotspotTransferActive = true;
        mWifiTransferActive = true;
        EventBus.getDefault().post(new EventMsg(EventMsg.MSG_TRANSFER_PROGRESS, -2, "热点回传：配网中..."));
        postLog("📡 正在开启手机热点...");
        try {
            wm.startLocalOnlyHotspot(new WifiManager.LocalOnlyHotspotCallback() {
                @Override
                @SuppressWarnings("deprecation")
                public void onStarted(WifiManager.LocalOnlyHotspotReservation reservation) {
                    mHotspotReservation = reservation;
                    try {
                        WifiConfiguration c = reservation.getWifiConfiguration();
                        mHotspotSsid = c != null ? c.SSID : null;
                        mHotspotPass = c != null ? c.preSharedKey : null;
                    } catch (Throwable e) {
                        Log.e(TAG, "hotspot config read error", e);
                    }
                    if (mHotspotSsid == null || mHotspotSsid.isEmpty()) {
                        postLog("⚠️ 热点凭证读取失败，回退 P2P");
                        stopHotspot();
                        startPhotoSync();
                        return;
                    }
                    postLog("📡 热点已开启 SSID=" + mHotspotSsid);
                    // BLE 配网：发两次确保送达
                    initCmdSender();
                    mCmdSender.sendWifiSTA(mHotspotSsid, mHotspotPass);
                    mMainHandler.postDelayed(() -> {
                        if (mHotspotTransferActive && mCmdSender != null) {
                            Log.i("GlassLog", "📤 [KSDK] 配网命令重发");
                            mCmdSender.sendWifiSTA(mHotspotSsid, mHotspotPass);
                        }
                    }, 1500);
                    // 轮询 ARP 等待眼镜接入
                    mHotspotScanRound = 0;
                    pollArpForGlasses();
                }

                @Override
                public void onFailed(int reason) {
                    postLog("⚠️ 热点启动失败 reason=" + reason + "，回退 P2P");
                    mHotspotTransferActive = false;
                    startPhotoSync();
                }
            }, mMainHandler);
        } catch (Exception e) {
            postLog("⚠️ 热点启动异常: " + e.getMessage() + "，回退 P2P");
            mHotspotTransferActive = false;
            startPhotoSync();
        }
    }

    /** 周期扫描热点网段，等待眼镜 STA 接入后探测其 HTTP 服务 */
    private void pollArpForGlasses() {
        if (!mHotspotTransferActive) return;
        mHotspotScanRound++;
        if (mHotspotScanRound > HOTSPOT_SCAN_ROUNDS) {
            postLog("⚠️ 热点模式 120s 未发现眼镜（配网可能失败），回退 P2P");
            stopHotspot();
            startPhotoSync();
            return;
        }
        new Thread(() -> {
            try {
                String ip = scanSubnetForGlasses();
                if (ip != null) {
                    postLog("🎯 热点模式发现眼镜 IP: " + ip);
                    // 注意：保留热点（传输正走热点网络），finishSync 时统一关闭
                    mMainHandler.post(() -> onGlassesIpObtained(ip));
                    return;
                }
            } catch (Throwable e) {
                Log.e(TAG, "subnet scan error", e);
            }
            mMainHandler.postDelayed(this::pollArpForGlasses, 2000);
        }).start();
    }

    /**
     * 主动并发扫描热点 /24 网段（被动等 ARP 表不可靠——手机不主动通信时眼镜不会出现在表里）。
     * 对网段内每个 IP 探测眼镜文件服务 /files/media.config，命中即返回。
     */
    private String scanSubnetForGlasses() {
        // 1. 收集本机所有 site-local IPv4 网段（去重；热点网段接口名各厂商不同，不按名字识别）
        java.util.LinkedHashSet<String> subnets = new java.util.LinkedHashSet<>();
        java.util.HashSet<String> selfIps = new java.util.HashSet<>();
        try {
            java.util.Enumeration<java.net.NetworkInterface> nifList =
                    java.net.NetworkInterface.getNetworkInterfaces();
            while (nifList.hasMoreElements()) {
                java.net.NetworkInterface nif = nifList.nextElement();
                if (!nif.isUp() || nif.isLoopback()) continue;
                java.util.Enumeration<java.net.InetAddress> addrs = nif.getInetAddresses();
                while (addrs.hasMoreElements()) {
                    java.net.InetAddress a = addrs.nextElement();
                    if (a instanceof java.net.Inet4Address && a.isSiteLocalAddress()) {
                        String ip = a.getHostAddress();
                        selfIps.add(ip);
                        subnets.add(ip.substring(0, ip.lastIndexOf('.') + 1));
                    }
                }
            }
        } catch (Exception ignored) {}
        if (subnets.isEmpty()) {
            postLog("🔍 [网段扫描] 无本地 IPv4 网段，跳过本轮");
            return null;
        }
        if (mHotspotScanRound % 10 == 1) {
            postLog("🔍 [网段扫描] 第" + mHotspotScanRound + "轮，候选网段=" + subnets
                    + " 本机IP=" + selfIps);
        }

        // 2. 逐网段并发探测
        for (String base : subnets) {
            java.util.concurrent.ExecutorService pool =
                    java.util.concurrent.Executors.newFixedThreadPool(32);
            java.util.List<java.util.concurrent.Future<String>> futures = new java.util.ArrayList<>();
            for (int i = 1; i <= 254; i++) {
                final String ip = base + i;
                if (selfIps.contains(ip)) continue;
                futures.add(pool.submit(() -> probeGlassesHttp(ip)));
            }
            String found = null;
            for (java.util.concurrent.Future<String> f : futures) {
                try {
                    String r = f.get(3, java.util.concurrent.TimeUnit.SECONDS);
                    if (r != null) { found = r; break; }
                } catch (Exception ignored) {}
            }
            pool.shutdown();
            if (found != null) return found;
        }
        return null;
    }

    /** 探测某 IP 是否为眼镜文件服务 */
    private String probeGlassesHttp(String ip) {
        try {
            java.net.HttpURLConnection conn = (java.net.HttpURLConnection)
                    new java.net.URL("http://" + ip + "/files/media.config").openConnection();
            conn.setConnectTimeout(600);
            conn.setReadTimeout(600);
            conn.setRequestMethod("GET");
            int code = conn.getResponseCode();
            conn.disconnect();
            if (code == 200) return ip; // 命中眼镜文件服务
        } catch (Exception ignored) {}
        return null;
    }

    /** 关闭热点并释放 reservation */
    private void stopHotspot() {
        if (mHotspotReservation != null) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                try { mHotspotReservation.close(); } catch (Exception ignored) {}
            }
            mHotspotReservation = null;
        }
    }

    public boolean isSingleShotActive() {
        return mSingleShotActive;
    }

    /** 停止连拍检测循环 */
    public void stopDetectionLoop() {
        if (!mDetectLoopActive) return;
        mDetectLoopActive = false;
        mMainHandler.removeCallbacks(mDetectNextRoundRunnable);
        mMainHandler.removeCallbacks(mDetectFallbackRunnable);
        mDetectNextScheduled = false;
        postLog("⏹️ 连拍检测循环已停止");
    }

    public boolean isDetectionLoopActive() {
        return mDetectLoopActive;
    }

    private void scheduleNextDetectRound() {
        if (!mDetectLoopActive || mDetectNextScheduled) return;
        mDetectNextScheduled = true;
        mMainHandler.postDelayed(mDetectNextRoundRunnable, DETECT_LOOP_INTERVAL_MS);
    }

    /** 对最新同步照片跑防松标记检测（新 YOLO 紧固件模型），结果（含预览图）通过 MSG_DETECT_RESULT 发往 UI。 */
    private void detectLatestPhotoWithYolo() {
        new Thread(() -> {
            try {
                if (!MarkedPointDetectorHolder.isReady(getApplicationContext())) {
                    String error = MarkedPointDetectorHolder.getInitializationError();
                    postLog("⚠️ 防松标记模型未就绪: " + error);
                    EventBus.getDefault().post(new EventMsg(EventMsg.MSG_DETECT_RESULT,
                            new DetectResult(error == null ? "防松标记模型加载失败" : error)));
                    return;
                }
                File latest = findLatestPhoto();
                if (latest == null) {
                    postLog("⚠️ 未找到照片，无法检测");
                    EventBus.getDefault().post(new EventMsg(EventMsg.MSG_DETECT_RESULT, new DetectResult("无照片")));
                    return;
                }
                Bitmap bitmap = decodeForDetection(latest.getAbsolutePath());
                if (bitmap == null) {
                    EventBus.getDefault().post(new EventMsg(EventMsg.MSG_DETECT_RESULT, new DetectResult("照片解码失败")));
                    return;
                }
                MarkedPointDetectorHolder.Result marked = MarkedPointDetectorHolder.detect(
                        getApplicationContext(), bitmap);
                List<YoloDetector.Detection> dets = marked.detections;
                long ms = Math.round(marked.latencyMillis);
                postLog("🎯 防松标记检测: " + dets.size() + " 个检查点, 推理 " + ms
                        + "ms (" + latest.getName() + ")");
                // bitmap 所有权交给 UI 作为预览图（UI 显示下一帧时回收）
                DetectResult result = new DetectResult(bitmap, dets, bitmap.getWidth(), bitmap.getHeight(), ms, latest.getName());
                EventBus.getDefault().post(new EventMsg(EventMsg.MSG_DETECT_RESULT, dets.size(), result));
            } catch (Throwable e) {
                Log.e(TAG, "detectLatestPhotoWithYolo error", e);
                postLog("⚠️ YOLO 检测异常: " + e.getMessage());
                EventBus.getDefault().post(new EventMsg(EventMsg.MSG_DETECT_RESULT, new DetectResult(e.getMessage())));
            }
        }).start();
    }

    /** 检测用解码：限制最长边 1280 足够 640 输入，减少内存与推理耗时 */
    private Bitmap decodeForDetection(String path) {
        try {
            BitmapFactory.Options opts = new BitmapFactory.Options();
            opts.inJustDecodeBounds = true;
            BitmapFactory.decodeFile(path, opts);
            int sample = 1;
            while (Math.max(opts.outWidth, opts.outHeight) / (sample * 2) >= 1280) sample *= 2;
            opts.inSampleSize = sample;
            opts.inJustDecodeBounds = false;
            opts.inPreferredConfig = Bitmap.Config.ARGB_8888;
            return BitmapFactory.decodeFile(path, opts);
        } catch (Exception e) {
            return null;
        }
    }

    /** 收到拍照事件（type=1）后自动触发同步（按拍照分流模式），带防抖和冷却期避免同步流程触发的事件导致死循环 */
    private void onPhotoCaptured(int seq) {
        // 序号为 0 表示模式切换（进入/退出导入模式），并非真实拍照，直接忽略
        if (seq == 0) {
            postLog("ℹ️ 忽略模式切换事件（序号=0）");
            return;
        }
        if (System.currentTimeMillis() < mAutoSyncCooldownUntil) {
            postLog("ℹ️ 冷却期内忽略拍照事件（序号=" + seq + "）");
            return;
        }
        postLog("📸 检测到拍照事件（照片序号=" + seq + "）");
        if (mSyncActive) {
            postLog("ℹ️ 正在同步中，忽略本次拍照事件");
            return;
        }
        // 非语音分流拍照（如眼镜按键）：仅自动导入原图库，不做识别
        if (mCaptureMode == CAPTURE_MODE_MANUAL) {
            mCaptureMode = CAPTURE_MODE_PLAIN;
        }
        mMainHandler.removeCallbacks(mAutoSyncRunnable);
        mMainHandler.postDelayed(mAutoSyncRunnable, 800);
    }

    /** 同步完成后，识别 photos 目录里最新一张照片中的二维码并发事件 */
    private void recognizeLatestPhoto() {
        new Thread(() -> {
            try {
                File latest = findLatestPhoto();
                if (latest == null) {
                    postLog("⚠️ 未找到照片，无法识别");
                    EventBus.getDefault().post(new EventMsg(EventMsg.MSG_QR_RESULT, ""));
                    return;
                }
                postLog("🔍 正在识别最新照片: " + latest.getName());
                Bitmap bitmap = decodeForRecognition(latest.getAbsolutePath());
                final String result = Vision.get().decodeQrCode(bitmap);
                if (bitmap != null) bitmap.recycle();
                if (result == null || result.isEmpty()) {
                    postLog("ℹ️ 最新照片中未识别到二维码");
                    EventBus.getDefault().post(new EventMsg(EventMsg.MSG_QR_RESULT, ""));
                } else {
                    postLog("✅ 识别到二维码: " + result);
                    EventBus.getDefault().post(new EventMsg(EventMsg.MSG_QR_RESULT, result));
                }
            } catch (Exception e) {
                Log.e(TAG, "recognizeLatestPhoto error", e);
                EventBus.getDefault().post(new EventMsg(EventMsg.MSG_QR_RESULT, ""));
            }
        }).start();
    }

    private File findLatestPhoto() {
        File dir = getPhotoDir();
        File[] files = dir.listFiles();
        if (files == null) return null;
        File latest = null;
        long max = 0;
        for (File f : files) {
            if (f.isFile() && f.lastModified() > max) {
                max = f.lastModified();
                latest = f;
            }
        }
        return latest;
    }

    private Bitmap decodeForRecognition(String path) {
        try {
            BitmapFactory.Options opts = new BitmapFactory.Options();
            opts.inJustDecodeBounds = true;
            BitmapFactory.decodeFile(path, opts);
            int reqSize = 2500;
            int sample = 1;
            int max = Math.max(opts.outWidth, opts.outHeight);
            while (max / sample > reqSize) sample *= 2;
            opts.inSampleSize = sample;
            opts.inJustDecodeBounds = false;
            opts.inPreferredConfig = Bitmap.Config.ARGB_8888;
            return BitmapFactory.decodeFile(path, opts);
        } catch (Exception e) {
            return null;
        }
    }

    private String[] splitLines(String s) {
        List<String> list = new LinkedList<>();
        for (String line : s.split("\\r?\\n")) {
            String t = line.trim();
            if (!t.isEmpty()) list.add(t);
        }
        return list.toArray(new String[0]);
    }

    @SuppressWarnings("ResultOfMethodCallIgnored")
    private File getPhotoDir() {
        File dir = new File(getExternalFilesDir(null), "glass_media/photos");
        if (!dir.exists()) dir.mkdirs();
        return dir;
    }

    private String httpGet(String urlStr) {
        HttpURLConnection conn = null;
        try {
            URL url = new URL(urlStr);
            conn = (HttpURLConnection) url.openConnection();
            conn.setConnectTimeout(8000);
            conn.setReadTimeout(8000);
            conn.setRequestMethod("GET");
            int code = conn.getResponseCode();
            if (code != 200) {
                postLog("⚠️ HTTP " + code + ": " + urlStr);
                return null;
            }
            InputStream is = conn.getInputStream();
            BufferedReader reader = new BufferedReader(new InputStreamReader(is));
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line).append('\n');
            }
            reader.close();
            return sb.toString();
        } catch (Exception e) {
            Log.e(TAG, "httpGet error " + urlStr, e);
            postLog("⚠️ HTTP请求失败: " + urlStr + " (" + e.getClass().getSimpleName() + ": " + e.getMessage() + ")");
            return null;
        } finally {
            if (conn != null) conn.disconnect();
        }
    }

    private boolean httpDownload(String urlStr, File outFile, String name) {
        HttpURLConnection conn = null;
        try {
            URL url = new URL(urlStr);
            conn = (HttpURLConnection) url.openConnection();
            conn.setConnectTimeout(10000);
            conn.setReadTimeout(30000);
            conn.setRequestMethod("GET");
            int code = conn.getResponseCode();
            if (code != 200) {
                postLog("⚠️ 下载失败(" + code + "): " + name);
                return false;
            }
            InputStream is = conn.getInputStream();
            FileOutputStream fos = new FileOutputStream(outFile);
            byte[] buf = new byte[8192];
            int len;
            while ((len = is.read(buf)) > 0) {
                fos.write(buf, 0, len);
            }
            fos.flush();
            fos.close();
            is.close();
            postLog("✅ 已保存: " + name);
            return true;
        } catch (Exception e) {
            Log.e(TAG, "httpDownload error " + urlStr, e);
            postLog("⚠️ 下载失败: " + name + " (" + e.getMessage() + ")");
            return false;
        } finally {
            if (conn != null) conn.disconnect();
        }
    }
}
