package com.ar.glass.ui;

import android.Manifest;
import android.app.Dialog;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothManager;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.PackageManager;
import android.location.LocationManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.provider.Settings;
import android.util.Log;
import android.util.LruCache;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import com.ar.glass.R;
import com.ar.glass.core.AppState;
import com.ar.glass.core.GlassBleService;
import com.ar.glass.record.MeterRecordStore;
import com.ar.glass.util.EventMsg;
import com.ar.glass.util.MeterTts;
import com.ar.glass.vision.MeterReading;
import com.ar.glass.vision.ThresholdAlarm;
import com.ar.glass.vision.ThresholdConfig;
import com.ar.glass.vision.Vision;
import com.ar.glass.vision.cloud.MeterCloudOcr;
import com.ar.glass.voice.VoiceController;

import org.greenrobot.eventbus.EventBus;
import org.greenrobot.eventbus.Subscribe;
import org.greenrobot.eventbus.ThreadMode;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private static final int PERMISSION_REQUEST_CODE = 100;
    private static final int CAMERA_REQUEST_CODE = 200;
    private static final String FILE_PROVIDER_AUTHORITY = "com.ar.glass.fileprovider";

    private TextView tvBleStatus;
    private TextView tvSystemStatus;
    private TextView tvDeviceName;
    private TextView tvBatteryStatus;
    private TextView tvLog;
    /** 日志区是否展开（默认收起：日志全量记录在后台，前台仅进度条） */
    private boolean mLogExpanded = false;
    private View cardTransfer;
    private TextView tvTransferStatus;
    private ProgressBar pbTransfer;

    private Button btnSyncPhotos;
    private Button btnGalleryOriginal;
    private Button btnSelectDevice;
    private Button btnVoice;
    private Button btnDetectLoop;
    private Button btnPhoneDetect;
    private TextView tvDetectStatus;
    private TextView tvDetectConf;
    private TextView tvDetectPlaceholder;
    private ImageView ivDetectPreview;
    private com.ar.glass.vision.ui.BoxOverlay detectOverlay;
    private SeekBar seekDetectConf;
    /** 当前显示的预览图（所有权在 Service/自测接收器 → UI，替换时回收旧图） */
    private Bitmap mDetectPreview;
    private Button btnRecords;
    private Button btnThreshold;
    private CheckBox cbVoice;
    private CheckBox cbAlarm;

    private ActivityResultLauncher<String> mPickImageLauncher;
    private ActivityResultLauncher<Uri> mTakePictureLauncher;
    private Uri mCaptureUri;
    private ActivityResultLauncher<Uri> mPhoneDetectLauncher;
    private Uri mPhoneDetectUri;
    private final ExecutorService mOcrExecutor = Executors.newSingleThreadExecutor();

    private MeterTts mTts;
    private MeterRecordStore mStore;
    private ThresholdConfig mThresholdConfig;

    private GlassBleService mBleService;
    private boolean mServiceBound = false;

    private VoiceController mVoiceController;

    private StringBuilder logBuilder = new StringBuilder();

    private ServiceConnection mServiceConnection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder service) {
            GlassBleService.LocalBinder binder = (GlassBleService.LocalBinder) service;
            mBleService = binder.getService();
            mServiceBound = true;
            appendLog("✅ BLE服务已绑定");
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
            mBleService = null;
            mServiceBound = false;
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initViews();
        initVoiceController();
        checkPermissions();
        EventBus.getDefault().register(this);

        mTts = new MeterTts(this);
        mStore = MeterRecordStore.get(this);
        mThresholdConfig = new ThresholdConfig(this);

        appendLog("AR眼镜控制应用启动...");
    }

    private void initViews() {
        tvBleStatus = findViewById(R.id.tvBleStatus);
        tvSystemStatus = findViewById(R.id.tvSystemStatus);
        tvDeviceName = findViewById(R.id.tvDeviceName);
        tvBatteryStatus = findViewById(R.id.tvBatteryStatus);
        tvLog = findViewById(R.id.tvLog);
        cardTransfer = findViewById(R.id.cardTransfer);
        tvTransferStatus = findViewById(R.id.tvTransferStatus);
        pbTransfer = findViewById(R.id.pbTransfer);
        // 日志区默认收起，点击标题展开/收起
        TextView tvLogToggle = findViewById(R.id.tvLogToggle);
        tvLogToggle.setOnClickListener(v -> {
            mLogExpanded = !mLogExpanded;
            tvLog.setVisibility(mLogExpanded ? View.VISIBLE : View.GONE);
            tvLogToggle.setText(mLogExpanded ? "▼ 运行日志（点击收起）" : "▶ 运行日志（点击展开）");
        });

        btnSyncPhotos = findViewById(R.id.btnSyncFiles);
        btnGalleryOriginal = findViewById(R.id.btnGalleryOriginal);
        btnSelectDevice = findViewById(R.id.btnSelectDevice);
        btnVoice = findViewById(R.id.btnVoice);
        btnDetectLoop = findViewById(R.id.btnDetectLoop);
        btnPhoneDetect = findViewById(R.id.btnPhoneDetect);
        tvDetectStatus = findViewById(R.id.tvDetectStatus);
        tvDetectConf = findViewById(R.id.tvDetectConf);
        tvDetectPlaceholder = findViewById(R.id.tvDetectPlaceholder);
        ivDetectPreview = findViewById(R.id.ivDetectPreview);
        detectOverlay = findViewById(R.id.detectOverlay);
        seekDetectConf = findViewById(R.id.seekDetectConf);

        // 紧固件/防松标记检测使用固定高召回阈值，不支持手动调节
        seekDetectConf.setEnabled(false);
        tvDetectConf.setText("固定高召回阈值");
        btnMeterRecognize = findViewById(R.id.btnMeterRecognize);
        btnCameraRecognize = findViewById(R.id.btnCameraRecognize);
        btnRecords = findViewById(R.id.btnRecords);
        btnThreshold = findViewById(R.id.btnThreshold);
        cbVoice = findViewById(R.id.cbVoice);
        cbAlarm = findViewById(R.id.cbAlarm);

        btnSyncPhotos.setOnClickListener(v -> syncPhotos());
        btnGalleryOriginal.setOnClickListener(v -> openGallery(GalleryActivity.MODE_ORIGINAL));
        btnSelectDevice.setOnClickListener(v -> showDeviceDialog());
        btnDetectLoop.setOnClickListener(v -> startSingleDetect());
        btnPhoneDetect.setOnClickListener(v -> launchPhoneDetectionCamera());
        btnVoice.setOnTouchListener((v, event) -> {
            if (mVoiceController == null) return false;
            switch (event.getAction()) {
                case MotionEvent.ACTION_DOWN:
                    appendLog("🎤 按住说话，松手识别...");
                    mVoiceController.startCapture();
                    return true;
                case MotionEvent.ACTION_UP:
                case MotionEvent.ACTION_CANCEL:
                    mVoiceController.stopCaptureAndRecognize();
                    return true;
            }
            return false;
        });

        btnRecords.setOnClickListener(v ->
                startActivity(new Intent(this, MeterRecordsActivity.class)));
        btnThreshold.setOnClickListener(v -> showThresholdDialog());

        // 万用表读数识别：现场拍照 → 云端识别
        mTakePictureLauncher = registerForActivityResult(
                new ActivityResultContracts.TakePicture(),
                success -> {
                    if (success && mCaptureUri != null) {
                        recognizeMeterFromUri(mCaptureUri);
                    }
                });
        btnCameraRecognize.setOnClickListener(v -> captureMeter());

        // 手机拍照检测：本机拍照 → 紧固件/防松标记检测
        mPhoneDetectLauncher = registerForActivityResult(
                new ActivityResultContracts.TakePicture(),
                success -> {
                    if (success && mPhoneDetectUri != null) {
                        detectPhonePhoto(mPhoneDetectUri);
                    } else {
                        resetPhoneDetectButton();
                    }
                });

        setControlsEnabled(false);
    }

    private void initVoiceController() {
        mVoiceController = new VoiceController(this, new VoiceController.Listener() {
            @Override
            public void onKeywordDetected(String keyword) {
                appendLog("🎤 识别到关键词: " + keyword);
                if (mBleService == null) {
                    appendLog("⚠️ BLE服务未就绪，无法拍照");
                    return;
                }
                // 语音分流：按识别文本选择拍照后的检测方式
                int mode = classifyCaptureMode(keyword);
                mBleService.takePhotoFor(mode);
            }

            @Override
            public void onSpeechText(String text) {
                appendLog("🎤 语音: " + text);
            }

            @Override
            public void onListeningChanged(boolean listening) {
                btnVoice.setText(listening ? "🔴 松开识别..." : "🎤 按住说话");
            }

            @Override
            public void onError(String message) {
                appendLog("⚠️ " + message);
                Toast.makeText(MainActivity.this, message, Toast.LENGTH_SHORT).show();
            }
        });
    }

    /**
     * 语音指令同音字容错表：ASR 常把指令词识别成同音/近音字（如“对齐”→“对其”）。
     * 每行为一条指令的全部变体，命中任意一个即视为该指令。
     * 新增指令时在此追加一行即可。
     */
    private static final String[][] VOICE_COMMAND_VARIANTS = {
            // “对齐”duì qí：qi 音同音字
            {"对齐", "对其", "对气", "对器", "对起", "对期", "对棋", "对企",
                    "对汽", "对砌", "对启", "对七", "对妻", "对旗", "对骑", "对祈"},
            // “二维码”èr wéi mǎ：wei 音同音字
            {"二维码", "而维码", "二唯码", "二围码", "尔维码", "2维码"},
            // “万用表”wàn yòng biǎo：yong 音同音字（“万能表”为常见口误）
            {"万用表", "万永表", "万庸表", "万勇表", "万能表", "婉用表"},
    };

    /** text 是否包含 variants 中的任意词 */
    private static boolean containsAny(String text, String[] variants) {
        for (String v : variants) {
            if (text.contains(v)) return true;
        }
        return false;
    }

    /**
     * 语音指令分类：
     * - “二维码拍照” → 拍照同步后识别二维码
     * - “对齐拍照” → 拍照同步后 YOLO 检测
     * - “万用表拍照” → 拍照同步后读数识别
     * - 普通拍照 → 仅同步存入原图库
     * 匹配时使用 {@link #VOICE_COMMAND_VARIANTS} 同音字容错。
     */
    private int classifyCaptureMode(String text) {
        if (text == null) return GlassBleService.CAPTURE_MODE_PLAIN;
        if (containsAny(text, VOICE_COMMAND_VARIANTS[1])) return GlassBleService.CAPTURE_MODE_QR;
        if (containsAny(text, VOICE_COMMAND_VARIANTS[0])) return GlassBleService.CAPTURE_MODE_YOLO;
        if (containsAny(text, VOICE_COMMAND_VARIANTS[2])) return GlassBleService.CAPTURE_MODE_METER;
        return GlassBleService.CAPTURE_MODE_PLAIN;
    }

    private void checkPermissions() {
        String[] permissions;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            permissions = new String[]{
                    Manifest.permission.BLUETOOTH_SCAN,
                    Manifest.permission.BLUETOOTH_CONNECT,
                    Manifest.permission.ACCESS_FINE_LOCATION,
                    Manifest.permission.ACCESS_COARSE_LOCATION,
                    Manifest.permission.READ_EXTERNAL_STORAGE,
                    Manifest.permission.RECORD_AUDIO,
                    Manifest.permission.INTERNET,
                    Manifest.permission.ACCESS_WIFI_STATE,
                    Manifest.permission.CHANGE_WIFI_STATE,
                    Manifest.permission.CHANGE_NETWORK_STATE,
                    Manifest.permission.ACCESS_NETWORK_STATE,
                    Manifest.permission.POST_NOTIFICATIONS
            };
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                permissions = new String[]{
                        Manifest.permission.BLUETOOTH_SCAN,
                        Manifest.permission.BLUETOOTH_CONNECT,
                        Manifest.permission.ACCESS_FINE_LOCATION,
                        Manifest.permission.ACCESS_COARSE_LOCATION,
                        Manifest.permission.READ_EXTERNAL_STORAGE,
                    Manifest.permission.RECORD_AUDIO,
                        Manifest.permission.INTERNET,
                        Manifest.permission.ACCESS_WIFI_STATE,
                        Manifest.permission.CHANGE_WIFI_STATE,
                        Manifest.permission.CHANGE_NETWORK_STATE,
                        Manifest.permission.ACCESS_NETWORK_STATE,
                        Manifest.permission.POST_NOTIFICATIONS,
                        Manifest.permission.NEARBY_WIFI_DEVICES
                };
            }
        } else {
            permissions = new String[]{
                    Manifest.permission.BLUETOOTH,
                    Manifest.permission.BLUETOOTH_ADMIN,
                    Manifest.permission.ACCESS_FINE_LOCATION,
                    Manifest.permission.ACCESS_COARSE_LOCATION,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE,
                    Manifest.permission.READ_EXTERNAL_STORAGE,
                    Manifest.permission.RECORD_AUDIO,
                    Manifest.permission.INTERNET,
                    Manifest.permission.ACCESS_WIFI_STATE,
                    Manifest.permission.CHANGE_WIFI_STATE,
                    Manifest.permission.CHANGE_NETWORK_STATE,
                    Manifest.permission.ACCESS_NETWORK_STATE
            };
        }

        boolean needRequest = false;
        for (String perm : permissions) {
            if (ContextCompat.checkSelfPermission(this, perm) != PackageManager.PERMISSION_GRANTED) {
                needRequest = true;
                break;
            }
        }

        if (needRequest) {
            ActivityCompat.requestPermissions(this, permissions, PERMISSION_REQUEST_CODE);
        } else {
            checkLocationAndStartService();
        }
    }

    private boolean isLocationEnabled() {
        LocationManager lm = (LocationManager) getSystemService(Context.LOCATION_SERVICE);
        if (lm == null) return false;
        boolean gpsEnabled = false;
        boolean networkEnabled = false;
        try {
            gpsEnabled = lm.isProviderEnabled(LocationManager.GPS_PROVIDER);
        } catch (Exception e) { /* ignore */ }
        try {
            networkEnabled = lm.isProviderEnabled(LocationManager.NETWORK_PROVIDER);
        } catch (Exception e) { /* ignore */ }
        return gpsEnabled || networkEnabled;
    }

    private void showLocationDialog() {
        new AlertDialog.Builder(this)
                .setTitle("⚠️ 需要开启位置服务")
                .setMessage("BLE蓝牙扫描需要开启手机的位置服务（GPS定位），否则无法搜索到眼镜设备。\n\n" +
                        "请点击「设置」开启位置服务。")
                .setPositiveButton("去设置", (dialog, which) -> {
                    try {
                        startActivity(new Intent(Settings.ACTION_LOCATION_SOURCE_SETTINGS));
                    } catch (Exception e) {
                        Toast.makeText(this, "请手动开启位置服务", Toast.LENGTH_LONG).show();
                    }
                })
                .setNegativeButton("已开启，重试", (dialog, which) -> checkLocationAndStartService())
                .setCancelable(false)
                .show();
    }

    private boolean isBluetoothEnabled() {
        try {
            BluetoothManager bm = (BluetoothManager) getSystemService(Context.BLUETOOTH_SERVICE);
            BluetoothAdapter adapter = bm != null ? bm.getAdapter() : null;
            return adapter != null && adapter.isEnabled();
        } catch (Exception e) {
            return false;
        }
    }

    private void showBluetoothDialog() {
        new AlertDialog.Builder(this)
                .setTitle("⚠️ 需要开启蓝牙")
                .setMessage("检测到手机蓝牙未开启，无法搜索到AR眼镜。\n\n请开启蓝牙，应用会自动继续搜索。")
                .setPositiveButton("去开启蓝牙", (dialog, which) -> {
                    try {
                        startActivity(new Intent(Settings.ACTION_BLUETOOTH_SETTINGS));
                    } catch (Exception e) {
                        Toast.makeText(this, "请手动开启蓝牙", Toast.LENGTH_LONG).show();
                    }
                })
                .setNegativeButton("稍后", (dialog, which) -> {})
                .show();
    }

    private void checkLocationAndStartService() {
        if (!isLocationEnabled()) {
            appendLog("⚠️ 位置服务未开启，BLE扫描可能找不到设备");
            showLocationDialog();
        } else {
            appendLog("✅ 位置服务已开启");
        }
        // 检查蓝牙，未开启则弹窗提示（不阻塞服务启动，服务会持续扫描，开蓝牙后自动搜到）
        if (!isBluetoothEnabled()) {
            appendLog("⚠️ 手机蓝牙未开启");
            showBluetoothDialog();
        }
        startAndBindBleService();
    }

    private void startAndBindBleService() {
        Intent intent = new Intent(this, GlassBleService.class);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent);
        } else {
            startService(intent);
        }
        bindService(intent, mServiceConnection, Context.BIND_AUTO_CREATE);
        appendLog("🔍 正在启动BLE服务，搜索眼镜设备...");
        appendLog("💡 提示：请确保眼镜已开机，手机蓝牙已打开");
        appendLog("💡 提示：BLE扫描需要开启「位置服务」，否则找不到设备");
        appendLog("💡 如果系统蓝牙已配对眼镜但App未连接，请在系统设置中取消配对后重试");
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQUEST_CODE) {
            boolean allGranted = true;
            for (int result : grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    allGranted = false;
                    break;
                }
            }
            if (allGranted) {
                appendLog("✅ 所有权限已授予");
            } else {
                appendLog("⚠️ 部分权限未授予，蓝牙/WiFi传输可能无法正常工作");
                Toast.makeText(this, "请授予所有权限以正常使用", Toast.LENGTH_LONG).show();
            }
            checkLocationAndStartService();
        } else if (requestCode == CAMERA_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                launchCamera();
            } else {
                Toast.makeText(this, "需要相机权限才能拍照识别", Toast.LENGTH_LONG).show();
            }
        }
    }

    @Subscribe(threadMode = ThreadMode.MAIN)
    public void onEvent(EventMsg msg) {
        switch (msg.what) {
            case EventMsg.MSG_CONNECT_STATE:
                boolean connected = msg.arg1 == 1;
                tvBleStatus.setText(connected ? "已连接" : "未连接");
                tvBleStatus.setTextColor(getColor(connected ? android.R.color.holo_green_dark : android.R.color.holo_red_dark));
                setControlsEnabled(connected && AppState.getInstance().isSystemReady);
                if (connected) {
                    tvDeviceName.setText(AppState.getInstance().bleName);
                    appendLog("✅ BLE已连接: " + AppState.getInstance().bleName);
                } else {
                    tvSystemStatus.setText("未就绪");
                    tvSystemStatus.setTextColor(getColor(android.R.color.holo_red_dark));
                    tvBatteryStatus.setText("-");
                    tvBatteryStatus.setTextColor(getColor(android.R.color.darker_gray));
                    appendLog("❌ BLE断开，正在重连...");
                    updateSyncButtonState(false);
                    // 断连时终止检测循环并复位按钮
                    resetDetectLoopUi();
                }
                break;

            case EventMsg.MSG_SYSTEM_READY:
                tvSystemStatus.setText("已就绪");
                tvSystemStatus.setTextColor(getColor(android.R.color.holo_green_dark));
                setControlsEnabled(true);
                appendLog("✅ 眼镜系统已就绪，可以开始同步照片");
                break;

            case EventMsg.MSG_WIFI_CONNECT_RESULT:
                boolean wifiSucc = msg.arg1 == 1;
                if (wifiSucc) {
                    appendLog("✅ WiFi已连接，正在建立文件传输通道...");
                    updateSyncButtonState(true);
                } else {
                    appendLog("❌ WiFi连接失败，请重试");
                    updateSyncButtonState(false);
                }
                break;

            case EventMsg.MSG_FILE_RECV_FINISH:
                String filePath = (String) msg.obj;
                if (filePath != null) {
                    String fn = new File(filePath).getName();
                    appendLog("✅ 已保存: " + fn);
                }
                break;

            case EventMsg.MSG_SYNC_COMPLETE:
                int count = msg.arg1;
                appendLog("🎉 同步完成！共接收 " + count + " 个文件");
                AppState.getInstance().isSocketConnected = false;
                updateSyncButtonState(false);
                break;

            case EventMsg.MSG_TOAST:
                String text = (String) msg.obj;
                if (text != null) {
                    Toast.makeText(this, text, Toast.LENGTH_SHORT).show();
                    appendLog("ℹ️ " + text);
                }
                break;

            case EventMsg.MSG_LOG:
                String logText = (String) msg.obj;
                if (logText != null) {
                    appendLog(logText);
                }
                break;

            case EventMsg.MSG_TRANSFER_PROGRESS: {
                int percent = msg.arg1;
                String info = msg.obj != null ? (String) msg.obj : "";
                if (percent == -1) {
                    // 结束：隐藏进度卡片
                    cardTransfer.setVisibility(View.GONE);
                } else {
                    cardTransfer.setVisibility(View.VISIBLE);
                    tvTransferStatus.setText(info);
                    if (percent == -2) {
                        // 不确定进度（配网/等待中）：转圈
                        pbTransfer.setIndeterminate(true);
                    } else {
                        pbTransfer.setIndeterminate(false);
                        pbTransfer.setProgress(percent);
                    }
                }
                break;
            }

            case EventMsg.MSG_PHOTO_LIST:
                Object listObj = msg.obj;
                if (listObj instanceof List) {
                    showPhotoSelectDialog((List<String>) listObj);
                }
                break;

            case EventMsg.MSG_QR_RESULT:
                String qrText = (String) msg.obj;
                showQrResultDialog(qrText);
                break;

            case EventMsg.MSG_METER_RECOGNIZE:
                // 语音“万用表拍照”分流：对同步到原图库的最新照片做云端读数识别
                String meterPath = (String) msg.obj;
                if (meterPath != null) {
                    recognizeMeterFromUri(Uri.fromFile(new File(meterPath)));
                }
                break;

            case EventMsg.MSG_DETECT_RESULT: {
                Object rObj = msg.obj;
                if (!(rObj instanceof com.ar.glass.vision.DetectResult)) break;
                renderDetectResult((com.ar.glass.vision.DetectResult) rObj);
                break;
            }

            case EventMsg.MSG_BATTERY_UPDATE:
                int battery = msg.arg1;
                boolean charging = msg.arg2 == 1;
                tvBatteryStatus.setText(battery + "%" + (charging ? " ⚡充电中" : ""));
                tvBatteryStatus.setTextColor(getColor(battery < 20
                        ? android.R.color.holo_red_dark : android.R.color.holo_green_dark));
                break;
        }
    }

    /** 渲染一轮 YOLO 检测结果：预览图 + 检测框叠加 + 语音播报（语音拍照与相册检测共用） */
    private void renderDetectResult(com.ar.glass.vision.DetectResult r) {
        if (!r.isSuccess()) {
            tvDetectStatus.setText("检测失败（" + r.error + "）");
            appendLog("⚠️ YOLO 检测失败: " + r.error);
            return;
        }
        String time = new java.text.SimpleDateFormat("HH:mm:ss", java.util.Locale.US)
                .format(new java.util.Date());
        int detections = r.detections != null ? r.detections.size() : 0;

        // 预览图：所有权交接，替换时回收旧图
        Bitmap old = mDetectPreview;
        mDetectPreview = r.preview;
        ivDetectPreview.setImageBitmap(r.preview);
        tvDetectPlaceholder.setVisibility(View.GONE);
        if (old != null && old != r.preview && !old.isRecycled()) old.recycle();

        // 检测框叠加（fitCenter 精确映射）
        detectOverlay.setResults(r.detections, r.frameW, r.frameH);

        tvDetectStatus.setText("[" + time + "] " + r.fileName + " · 检测到 "
                + detections + " 个目标 · 推理 " + r.inferMs + "ms");
        appendLog("🎯 [" + time + "] YOLO 检测到 " + detections + " 个目标");
        // 结果通过眼镜扬声器语音播报（TTS 走 A2DP 媒体通道）
        if (mVoiceController != null) {
            mVoiceController.speak(detections > 0
                    ? ("检测到 " + detections + " 个目标")
                    : "未检测到目标");
        }
    }

    /** BLE 断开时复位检测状态提示 */
    private void resetDetectLoopUi() {
        if (tvDetectStatus != null) {
            tvDetectStatus.setText("蓝牙已断开，请重新连接后再拍照检测");
        }
    }

    /** 单张检测：拍照 → 同步一张 → YOLO → 预览（连拍逻辑已停用） */
    private void startSingleDetect() {
        Log.i("GlassLog", "🔘 [UI] 用户点击单张检测: btnEnabled=" + btnDetectLoop.isEnabled()
                + " bleConnected=" + AppState.getInstance().isBleConnected
                + " service=" + (mBleService != null)
                + " singleActive=" + (mBleService != null && mBleService.isSingleShotActive()));
        if (mBleService == null) {
            Log.i("GlassLog", "🔘 [UI] 拒绝: BLE服务未绑定");
            Toast.makeText(this, "BLE服务未就绪", Toast.LENGTH_SHORT).show();
            return;
        }
        if (!AppState.getInstance().isBleConnected) {
            Log.i("GlassLog", "🔘 [UI] 拒绝: 蓝牙未连接眼镜");
            Toast.makeText(this, "请等待蓝牙连接眼镜", Toast.LENGTH_SHORT).show();
            return;
        }
        btnDetectLoop.setEnabled(false);
        btnDetectLoop.setText("⏳ 拍照同步检测中…");
        tvDetectStatus.setText("流程：眼镜拍照 → 传输到手机 → 防松标记检测 → 显示预览");
        mBleService.startSingleShotDetection();
        Log.i("GlassLog", "🔘 [UI] 服务已接受单张检测: singleActive=" + mBleService.isSingleShotActive());
        if (!mBleService.isSingleShotActive()) {
            // 服务拒绝启动（如上一张同步中），立即复位按钮，避免永久禁用
            Log.i("GlassLog", "🔘 [UI] 服务拒绝启动，复位按钮");
            resetSingleDetectButton();
        }
    }

    /** 单张检测完成后复位按钮 */
    private void resetSingleDetectButton() {
        btnDetectLoop.setEnabled(true);
        btnDetectLoop.setText("👓 眼镜拍照检测");
    }

    /** 手机独立检测入口：无需连接眼镜，调用系统相机拍原图。 */
    private void launchPhoneDetectionCamera() {
        try {
            File captureFile = new File(getCacheDir(),
                    "fastener_capture_" + System.currentTimeMillis() + ".jpg");
            mPhoneDetectUri = FileProvider.getUriForFile(
                    this, FILE_PROVIDER_AUTHORITY, captureFile);
            btnPhoneDetect.setEnabled(false);
            btnPhoneDetect.setText("📷 等待手机拍照…");
            mPhoneDetectLauncher.launch(mPhoneDetectUri);
        } catch (Exception e) {
            Log.e(TAG, "启动手机检测相机失败", e);
            resetPhoneDetectButton();
            Toast.makeText(this, "无法启动手机相机：" + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private void detectPhonePhoto(Uri uri) {
        btnPhoneDetect.setText("⏳ 正在检测防松标记…");
        tvDetectStatus.setText("手机照片已获取，正在执行全图防松标记检测");
        mOcrExecutor.execute(() -> {
            Bitmap bitmap = null;
            try {
                byte[] bytes = readBytesFromUri(uri);
                bitmap = bytes == null ? null : decodeBytes(bytes, 2560);
                if (bitmap == null) {
                    throw new IllegalStateException("手机照片解码失败");
                }
                com.ar.glass.vision.MarkedPointDetectorHolder.Result marked =
                        com.ar.glass.vision.MarkedPointDetectorHolder.detect(this, bitmap);
                long elapsed = Math.round(marked.latencyMillis);
                EventBus.getDefault().post(new EventMsg(
                        EventMsg.MSG_DETECT_RESULT,
                        marked.detections.size(),
                        new com.ar.glass.vision.DetectResult(
                                bitmap,
                                marked.detections,
                                bitmap.getWidth(),
                                bitmap.getHeight(),
                                elapsed,
                                "手机拍照")));
                bitmap = null; // ownership transferred to the preview UI
            } catch (Throwable error) {
                if (bitmap != null && !bitmap.isRecycled()) bitmap.recycle();
                EventBus.getDefault().post(new EventMsg(
                        EventMsg.MSG_DETECT_RESULT,
                        new com.ar.glass.vision.DetectResult(
                                error.getMessage() == null ? "手机照片检测失败" : error.getMessage())));
            } finally {
                runOnUiThread(this::resetPhoneDetectButton);
            }
        });
    }

    private void resetPhoneDetectButton() {
        if (btnPhoneDetect != null) {
            btnPhoneDetect.setEnabled(true);
            btnPhoneDetect.setText("📱 手机拍照检测");
        }
    }

    /**
     * 一键同步照片（自动开热点→自动连WiFi→自动传文件）
     */
    private void syncPhotos() {
        if (!AppState.getInstance().isBleConnected) {
            Toast.makeText(this, "请等待蓝牙连接眼镜", Toast.LENGTH_SHORT).show();
            return;
        }
        if (!AppState.getInstance().isSystemReady) {
            Toast.makeText(this, "眼镜系统尚未就绪，请稍候", Toast.LENGTH_SHORT).show();
            return;
        }
        btnSyncPhotos.setEnabled(false);
        btnSyncPhotos.setText("同步中...");
        appendLog("🔄 开始同步照片...");
        if (mBleService != null) {
            mBleService.syncPhotos();
        }
        // 15秒后恢复按钮（防止一直卡住）
        btnSyncPhotos.postDelayed(() -> {
            if (!AppState.getInstance().isSocketConnected) {
                btnSyncPhotos.setEnabled(true);
                btnSyncPhotos.setText("同步照片到手机");
            }
        }, 15000);
    }

    private void updateSyncButtonState(boolean connecting) {
        btnSyncPhotos.setEnabled(true);
        if (connecting || AppState.getInstance().isSocketConnected) {
            btnSyncPhotos.setText("📤 传输中...");
            btnSyncPhotos.setEnabled(false);
        } else {
            btnSyncPhotos.setText("同步照片到手机");
        }
    }

    /** 弹窗显示二维码识别结果 */
    private void showQrResultDialog(String qrText) {
        if (qrText == null || qrText.isEmpty()) {
            if (mVoiceController != null) {
                mVoiceController.speak("未识别到二维码");
            }
            new AlertDialog.Builder(this)
                    .setTitle("识别结果")
                    .setMessage("未在最新照片中识别到二维码")
                    .setPositiveButton("确定", null)
                    .show();
        } else {
            if (mVoiceController != null) {
                mVoiceController.speak("二维码内容 " + qrText);
            }
            new AlertDialog.Builder(this)
                    .setTitle("二维码内容")
                    .setMessage(qrText)
                    .setPositiveButton("确定", null)
                    .show();
        }
    }

    private void openGallery(String mode) {
        Intent intent = new Intent(this, GalleryActivity.class);
        intent.putExtra(GalleryActivity.EXTRA_MODE, mode);
        startActivity(intent);
    }

    private void recognizeMeterFromUri(Uri uri) {
        appendLog("🔍 正在云端识别万用表读数与挡位，请稍候...");

        mOcrExecutor.execute(() -> {
            byte[] originalBytes = readBytesFromUri(uri);
            Bitmap bitmap = originalBytes != null ? decodeBytes(originalBytes, 2048) : null;

            MeterReading reading = null;
            String error = null;
            if (bitmap != null) {
                try {
                    reading = Vision.get().readMeter(bitmap);
                } catch (Exception e) {
                    error = "识别异常：" + e.getMessage();
                }
                if ((reading == null || !reading.hasValue()) && error == null) {
                    error = MeterCloudOcr.getLastError();
                }
            } else {
                error = "图片解码失败";
            }

            // 识别成功 → 自动保存到巡检台账（含原照片）
            if (reading != null && reading.hasValue()) {
                mStore.add(reading, originalBytes);
            }

            final MeterReading fReading = reading;
            final String fError = error;
            runOnUiThread(() -> {
                if (fReading != null && fReading.hasValue()) {
                    String display = fReading.getDisplayText();
                    String log = "✅ 识别结果: " + display
                            + (fReading.gear.isEmpty() ? "" : "（挡位: " + fReading.gear + "）")
                            + "，已保存到台账";
                    if (cbAlarm != null && cbAlarm.isChecked()) {
                        ThresholdAlarm.Result alarm = ThresholdAlarm.check(fReading, mThresholdConfig);
                        if (alarm != null && alarm.isAlarm()) {
                            log += "\n🚨 " + ThresholdAlarm.describe(alarm, fReading);
                        }
                    }
                    appendLog(log);
                    showMeterResultDialog(fReading);
                    speakMeterReading(fReading);
                } else {
                    String msg = fError != null ? fError : "未识别到读数";
                    appendLog("❌ " + msg);
                    new AlertDialog.Builder(this)
                            .setTitle("🔍 万用表读数")
                            .setMessage("未识别到读数。\n\n" + msg
                                    + "\n\n请确认：1) 图片为万用表/电压表屏幕照片；2) 数字清晰可见；3) 网络正常")
                            .setPositiveButton("确定", null)
                            .show();
                }
            });
        });
    }

    /** 展示识别结果：读数 + 挡位 + 异常提示 + 阈值报警 + 保存状态。 */
    private void showMeterResultDialog(MeterReading r) {
        // 阈值报警判断
        ThresholdAlarm.Result alarm = null;
        if (cbAlarm != null && cbAlarm.isChecked()) {
            alarm = ThresholdAlarm.check(r, mThresholdConfig);
        }

        StringBuilder msg = new StringBuilder();
        msg.append("读数：").append(r.getDisplayText());
        if (r.gear != null && !r.gear.isEmpty()) {
            msg.append("\n挡位：").append(r.gear);
        }
        if (r.unit == null || r.unit.isEmpty()) {
            String inferred = r.inferUnitFromGear();
            if (!inferred.isEmpty()) {
                msg.append("\n（单位按挡位推断为：").append(inferred).append("）");
            }
        }
        if (r.warning != null && !r.warning.isEmpty()) {
            msg.append("\n\n⚠️ ").append(r.warning);
        }
        if (alarm != null && alarm.isAlarm()) {
            msg.append("\n\n🚨 ").append(ThresholdAlarm.describe(alarm, r));
        }
        msg.append("\n\n✅ 已自动保存到巡检台账");

        AlertDialog.Builder b = new AlertDialog.Builder(this)
                .setTitle("🔍 万用表读数")
                .setMessage(msg.toString())
                .setPositiveButton("确定", null);
        b.setNegativeButton("查看台账", (d, w) ->
                startActivity(new Intent(this, MeterRecordsActivity.class)));
        b.show();
    }

    /** 阈值设置弹窗：按类别选择 → 配置上下限。 */
    private void showThresholdDialog() {
        final String[] labels = new String[ThresholdConfig.CATEGORIES.length];
        for (int i = 0; i < ThresholdConfig.CATEGORIES.length; i++) {
            ThresholdConfig.Bounds b = mThresholdConfig.getBounds(ThresholdConfig.CATEGORIES[i]);
            labels[i] = ThresholdConfig.CATEGORIES[i] + describeBounds(b);
        }
        new AlertDialog.Builder(this)
                .setTitle("⚙️ 阈值设置\n（选择量纲类别，读数超出上下限即报警）")
                .setItems(labels, (d, w) -> showCategoryBoundsDialog(ThresholdConfig.CATEGORIES[w]))
                .setNegativeButton("关闭", null)
                .show();
    }

    /** 某类别上下限的简要描述，如 "（上限 36.0）"。 */
    private String describeBounds(ThresholdConfig.Bounds b) {
        if (b == null || (!b.upperEnabled && !b.lowerEnabled)) {
            return "（未设置）";
        }
        StringBuilder sb = new StringBuilder("（");
        if (b.upperEnabled) {
            sb.append("上限 ").append(formatNum(b.upper));
        }
        if (b.lowerEnabled) {
            if (b.upperEnabled) sb.append("，");
            sb.append("下限 ").append(formatNum(b.lower));
        }
        sb.append("）");
        return sb.toString();
    }

    /** 配置某类别的上下限。 */
    private void showCategoryBoundsDialog(final String category) {
        ThresholdConfig.Bounds cur = mThresholdConfig.getBounds(category);

        LinearLayout container = new LinearLayout(this);
        container.setOrientation(LinearLayout.VERTICAL);
        int pad = dp(20);
        container.setPadding(pad, dp(16), pad, dp(4));

        final CheckBox cbUpper = new CheckBox(this);
        cbUpper.setText("启用上限");
        cbUpper.setChecked(cur.upperEnabled);
        container.addView(cbUpper);

        final EditText etUpper = new EditText(this);
        etUpper.setHint("上限值（如 36.0）");
        etUpper.setInputType(android.text.InputType.TYPE_CLASS_NUMBER
                | android.text.InputType.TYPE_NUMBER_FLAG_DECIMAL
                | android.text.InputType.TYPE_NUMBER_FLAG_SIGNED);
        if (cur.upperEnabled && !Double.isNaN(cur.upper)) {
            etUpper.setText(formatNum(cur.upper));
        }
        container.addView(etUpper);

        final CheckBox cbLower = new CheckBox(this);
        cbLower.setText("启用下限");
        cbLower.setChecked(cur.lowerEnabled);
        cbLower.setPadding(0, dp(12), 0, 0);
        container.addView(cbLower);

        final EditText etLower = new EditText(this);
        etLower.setHint("下限值（如 0.0）");
        etLower.setInputType(android.text.InputType.TYPE_CLASS_NUMBER
                | android.text.InputType.TYPE_NUMBER_FLAG_DECIMAL
                | android.text.InputType.TYPE_NUMBER_FLAG_SIGNED);
        if (cur.lowerEnabled && !Double.isNaN(cur.lower)) {
            etLower.setText(formatNum(cur.lower));
        }
        container.addView(etLower);

        new AlertDialog.Builder(this)
                .setTitle(category + " 阈值")
                .setView(container)
                .setPositiveButton("保存", (d, w) -> {
                    boolean upperOn = cbUpper.isChecked();
                    double upper = parseNum(etUpper.getText().toString());
                    boolean lowerOn = cbLower.isChecked();
                    double lower = parseNum(etLower.getText().toString());
                    if (upperOn && Double.isNaN(upper)) {
                        Toast.makeText(this, "请填写有效的上限值", Toast.LENGTH_SHORT).show();
                        return;
                    }
                    if (lowerOn && Double.isNaN(lower)) {
                        Toast.makeText(this, "请填写有效的下限值", Toast.LENGTH_SHORT).show();
                        return;
                    }
                    if (upperOn && lowerOn && upper < lower) {
                        Toast.makeText(this, "上限不能小于下限", Toast.LENGTH_SHORT).show();
                        return;
                    }
                    mThresholdConfig.setBounds(category, upperOn, upper, lowerOn, lower);
                    Toast.makeText(this, "已保存 " + category + " 阈值", Toast.LENGTH_SHORT).show();
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private static String formatNum(double d) {
        if (Double.isNaN(d)) return "";
        if (d == (long) d) return String.valueOf((long) d);
        return String.valueOf(d);
    }

    private static double parseNum(String s) {
        if (s == null) return Double.NaN;
        String t = s.trim();
        if (t.isEmpty()) return Double.NaN;
        try {
            return Double.parseDouble(t);
        } catch (Exception e) {
            return Double.NaN;
        }
    }

    private int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density + 0.5f);
    }

    /** 语音播报识别结果（开关打开且 TTS 就绪时）。 */
    private void speakMeterReading(MeterReading r) {
        if (cbVoice == null || !cbVoice.isChecked() || mTts == null) {
            return;
        }
        String text = r.getSpeechText();
        if (cbAlarm != null && cbAlarm.isChecked()) {
            ThresholdAlarm.Result alarm = ThresholdAlarm.check(r, mThresholdConfig);
            if (alarm != null && alarm.isAlarm()) {
                text = text + "，" + ThresholdAlarm.speechText(alarm);
            }
        }
        mTts.speak(text);
    }

    /**
     * 从 Uri 读取原始图片字节。
     */
    private byte[] readBytesFromUri(Uri uri) {
        try {
            InputStream is = getContentResolver().openInputStream(uri);
            if (is == null) {
                return null;
            }
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            byte[] buf = new byte[8192];
            int n;
            while ((n = is.read(buf)) != -1) {
                baos.write(buf, 0, n);
            }
            is.close();
            byte[] bytes = baos.toByteArray();
            return bytes.length == 0 ? null : bytes;
        } catch (Exception e) {
            Log.e(TAG, "读取图片失败", e);
            return null;
        }
    }

    /**
     * 按采样率解码字节为 Bitmap，限制最长边避免 OOM。
     */
    private Bitmap decodeBytes(byte[] bytes, int maxEdge) {
        if (bytes == null || bytes.length == 0) {
            return null;
        }
        try {
            BitmapFactory.Options opts = new BitmapFactory.Options();
            opts.inJustDecodeBounds = true;
            BitmapFactory.decodeByteArray(bytes, 0, bytes.length, opts);
            int sample = 1;
            int maxDim = Math.max(opts.outWidth, opts.outHeight);
            while (maxDim / sample > maxEdge) {
                sample *= 2;
            }
            opts.inJustDecodeBounds = false;
            opts.inSampleSize = sample;
            opts.inPreferredConfig = Bitmap.Config.ARGB_8888;
            return BitmapFactory.decodeByteArray(bytes, 0, bytes.length, opts);
        } catch (Exception e) {
            Log.e(TAG, "解码图片失败", e);
            return null;
        }
    }

    /** 显示眼镜设备列表，按「已连接 / 已配对 / 已发现」分区展示，供用户选择连接 */
    private void showDeviceDialog() {
        if (mBleService == null) {
            Toast.makeText(this, "BLE服务未就绪，请稍候", Toast.LENGTH_SHORT).show();
            return;
        }

        // 已连接：仅当前正在连接的眼镜
        final List<GlassBleService.DeviceInfo> connected = new ArrayList<>();
        GlassBleService.DeviceInfo current = mBleService.getConnectedDevice();
        if (current != null) connected.add(current);

        // 已配对：系统历史配对过的所有眼镜
        final List<GlassBleService.DeviceInfo> paired = mBleService.getPairedDevices();

        // 已发现：扫描到但尚未配对的眼镜（用于首次连接）
        final List<GlassBleService.DeviceInfo> discovered = mBleService.getDiscoveredDevices();
        final List<GlassBleService.DeviceInfo> newDevices = new ArrayList<>();
        for (GlassBleService.DeviceInfo d : discovered) {
            boolean alreadyPaired = false;
            for (GlassBleService.DeviceInfo p : paired) {
                if (d.address.equals(p.address)) { alreadyPaired = true; break; }
            }
            if (!alreadyPaired) newDevices.add(d);
        }

        if (connected.isEmpty() && paired.isEmpty() && newDevices.isEmpty()) {
            Toast.makeText(this, "暂未发现眼镜，正在扫描中，请稍候再试", Toast.LENGTH_SHORT).show();
            appendLog("ℹ️ 暂未发现眼镜设备，继续扫描中...");
            return;
        }

        float density = getResources().getDisplayMetrics().density;
        ScrollView scroll = new ScrollView(this);
        LinearLayout container = new LinearLayout(this);
        container.setOrientation(LinearLayout.VERTICAL);
        container.setPadding((int) (16 * density), (int) (16 * density),
                (int) (16 * density), (int) (16 * density));
        scroll.addView(container);

        final AlertDialog[] dialogHolder = new AlertDialog[1];

        // 已连接
        addSectionHeader(container, "已连接");
        if (connected.isEmpty()) {
            addDeviceRow(container, "（当前未连接眼镜）", null, false);
        } else {
            for (GlassBleService.DeviceInfo d : connected) {
                addDeviceRow(container, d.name + "  (" + d.address + ")", null, false);
            }
        }

        // 已配对（历史配对记录）
        addSectionHeader(container, "已配对（历史配对记录）");
        if (paired.isEmpty()) {
            addDeviceRow(container, "（无历史配对记录）", null, false);
        } else {
            for (GlassBleService.DeviceInfo d : paired) {
                addDeviceRow(container, d.name + "  (" + d.address + ")", () -> {
                    if (dialogHolder[0] != null) dialogHolder[0].dismiss();
                    connectSelectedDevice(d);
                }, true);
            }
        }

        // 已发现（未配对的新设备）
        if (!newDevices.isEmpty()) {
            addSectionHeader(container, "已发现（未配对）");
            for (GlassBleService.DeviceInfo d : newDevices) {
                addDeviceRow(container, d.name + "  (" + d.address + ")", () -> {
                    if (dialogHolder[0] != null) dialogHolder[0].dismiss();
                    connectSelectedDevice(d);
                }, true);
            }
        }

        dialogHolder[0] = new AlertDialog.Builder(this)
                .setTitle("选择眼镜设备")
                .setView(scroll)
                .setNegativeButton("刷新", (dialog, which) -> showDeviceDialog())
                .setPositiveButton("关闭", null)
                .show();
    }

    /** 向设备分区容器添加标题行 */
    private void addSectionHeader(LinearLayout container, String title) {
        TextView tv = new TextView(this);
        tv.setText(title);
        tv.setTextSize(14);
        tv.setTypeface(Typeface.DEFAULT_BOLD);
        tv.setTextColor(0xFF2196F3);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.topMargin = (int) (12 * getResources().getDisplayMetrics().density);
        container.addView(tv, lp);
    }

    /** 向设备分区容器添加一行设备信息 */
    private void addDeviceRow(LinearLayout container, String text, final Runnable onClick, boolean clickable) {
        TextView tv = new TextView(this);
        tv.setText(text);
        tv.setTextSize(14);
        tv.setPadding(0, (int) (10 * getResources().getDisplayMetrics().density),
                0, (int) (10 * getResources().getDisplayMetrics().density));
        if (clickable) {
            tv.setTextColor(0xFF1565C0);
            tv.setOnClickListener(v -> onClick.run());
        } else {
            tv.setTextColor(0xFF333333);
        }
        container.addView(tv);
    }

    private void connectSelectedDevice(GlassBleService.DeviceInfo d) {
        appendLog("🔗 选择连接: " + d.name);
        if (mBleService != null) mBleService.connectToDevice(d.address, d.name);
    }

    /** 显示照片勾选列表（带缩略图），用户选择要导入的图片 */
    private void showPhotoSelectDialog(List<String> names) {
        if (names == null || names.isEmpty()) {
            Toast.makeText(this, "眼镜中没有照片", Toast.LENGTH_SHORT).show();
            appendLog("ℹ️ 眼镜中没有可导入的照片");
            updateSyncButtonState(false);
            return;
        }

        File tmpDir = new File(getExternalFilesDir(null), "glass_media/tmp");
        final Dialog dialog = new Dialog(this);
        View content = getLayoutInflater().inflate(R.layout.dialog_photo_select, null);
        dialog.setContentView(content);

        ListView lvPhotos = content.findViewById(R.id.lvPhotos);
        Button btnSelectAll = content.findViewById(R.id.btnSelectAll);
        Button btnImportSelected = content.findViewById(R.id.btnImportSelected);
        Button btnCancelImport = content.findViewById(R.id.btnCancelImport);

        final List<String> fullList = new ArrayList<>(names);
        final boolean[] checked = new boolean[fullList.size()];
        for (int i = 0; i < checked.length; i++) checked[i] = true; // 默认全选

        final PhotoSelectAdapter adapter = new PhotoSelectAdapter(this, fullList, checked, tmpDir);
        lvPhotos.setAdapter(adapter);

        final boolean[] allSelected = {true};
        btnSelectAll.setText("全不选"); // 初始已全选
        btnSelectAll.setOnClickListener(v -> {
            allSelected[0] = !allSelected[0];
            for (int i = 0; i < checked.length; i++) checked[i] = allSelected[0];
            adapter.notifyDataSetChanged();
            btnSelectAll.setText(allSelected[0] ? "全不选" : "全选");
        });

        btnImportSelected.setOnClickListener(v -> {
            List<String> selected = new ArrayList<>();
            for (int i = 0; i < fullList.size(); i++) {
                if (checked[i]) selected.add(fullList.get(i));
            }
            dialog.dismiss();
            if (selected.isEmpty()) {
                Toast.makeText(this, "未选择任何照片", Toast.LENGTH_SHORT).show();
                if (mBleService != null) mBleService.cancelSync();
                return;
            }
            appendLog("📥 导入 " + selected.size() + " 张照片...");
            btnSyncPhotos.setText("📤 传输中...");
            btnSyncPhotos.setEnabled(false);
            if (mBleService != null) mBleService.finalizeImport(selected);
        });

        btnCancelImport.setOnClickListener(v -> {
            dialog.dismiss();
            if (mBleService != null) mBleService.cancelSync();
        });

        dialog.setOnDismissListener(d -> adapter.shutdown());
        dialog.setCancelable(false);
        dialog.show();
    }

    private void setControlsEnabled(boolean enabled) {
        btnSyncPhotos.setEnabled(enabled);
        if (enabled) {
            btnSyncPhotos.setText("同步照片到手机");
        }
    }

    private void appendLog(String text) {
        // 日志已由 GlassBleService.postLog 写入 logcat，此处仅 UI 展示（避免重复）
        logBuilder.append(text).append("\n");
        // 限制内存中的日志量（保留最近 ~400 行，全量日志走 logcat）
        int nl = 0, cut = 0;
        String s = logBuilder.toString();
        for (int i = s.length() - 1; i >= 0 && nl <= 400; i--) {
            if (s.charAt(i) == '\n') nl++;
            cut = i;
        }
        if (nl > 400) logBuilder.delete(0, cut + 1);
        // 仅展开时刷新前台日志区（默认收起，避免刷屏）
        if (mLogExpanded && tvLog != null) {
            tvLog.setText(logBuilder.toString());
            tvLog.post(() -> {
                final int lineCount = tvLog.getLineCount();
                if (lineCount > 0 && tvLog.getLayout() != null) {
                    int scrollY = tvLog.getLayout().getLineTop(lineCount) - tvLog.getHeight();
                    if (scrollY > 0) {
                        tvLog.scrollTo(0, scrollY);
                    }
                }
            });
        }
        Log.d(TAG, text);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (AppState.getInstance().isSocketConnected) {
            updateSyncButtonState(true);
        } else {
            updateSyncButtonState(false);
        }
        if (!isLocationEnabled() && !AppState.getInstance().isBleConnected) {
            appendLog("⚠️ 位置服务似乎已关闭，请开启以确保BLE扫描正常");
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mServiceBound) {
            unbindService(mServiceConnection);
            mServiceBound = false;
        }
        if (mDetectPreview != null && !mDetectPreview.isRecycled()) {
            mDetectPreview.recycle();
        }
        mDetectPreview = null;
        if (detectOverlay != null) detectOverlay.clear();
        if (mVoiceController != null) {
            mVoiceController.release();
        }
        mOcrExecutor.shutdownNow();
        if (mTts != null) {
            mTts.shutdown();
        }
        EventBus.getDefault().unregister(this);
    }

    /**
     * 照片选择适配器：显示缩略图 + 勾选框 + 文件名
     */
    private class PhotoSelectAdapter extends BaseAdapter {
        private final List<String> names;
        private final boolean[] checked;
        private final File tmpDir;
        private final LayoutInflater inflater;
        private final ExecutorService executor = Executors.newFixedThreadPool(3);
        private final LruCache<String, Bitmap> thumbCache;

        PhotoSelectAdapter(Context context, List<String> names, boolean[] checked, File tmpDir) {
            this.names = names;
            this.checked = checked;
            this.tmpDir = tmpDir;
            this.inflater = LayoutInflater.from(context);
            final int maxMemory = (int) (Runtime.getRuntime().maxMemory() / 1024);
            this.thumbCache = new LruCache<String, Bitmap>(maxMemory / 8) {
                @Override
                protected int sizeOf(String key, Bitmap bitmap) {
                    return bitmap.getByteCount() / 1024;
                }
            };
        }

        @Override
        public int getCount() {
            return names.size();
        }

        @Override
        public Object getItem(int position) {
            return names.get(position);
        }

        @Override
        public long getItemId(int position) {
            return position;
        }

        @Override
        public View getView(int position, View convertView, ViewGroup parent) {
            ViewHolder holder;
            if (convertView == null) {
                convertView = inflater.inflate(R.layout.item_photo_select, parent, false);
                holder = new ViewHolder();
                holder.cbSelect = convertView.findViewById(R.id.cbSelect);
                holder.ivThumb = convertView.findViewById(R.id.ivThumb);
                holder.tvFileName = convertView.findViewById(R.id.tvFileName);
                convertView.setTag(holder);
            } else {
                holder = (ViewHolder) convertView.getTag();
            }

            final String name = names.get(position);
            holder.tvFileName.setText(name);

            final int pos = position;
            final CheckBox cb = holder.cbSelect;
            cb.setChecked(checked[position]);
            cb.setOnClickListener(v -> checked[pos] = cb.isChecked());

            final ImageView iv = holder.ivThumb;
            final File imgFile = new File(tmpDir, name);
            final String path = imgFile.getAbsolutePath();
            iv.setTag(path);
            iv.setImageResource(android.R.color.darker_gray);

            Bitmap cached = thumbCache.get(path);
            if (cached != null && !cached.isRecycled()) {
                iv.setImageBitmap(cached);
            } else {
                executor.execute(() -> {
                    Bitmap thumb = decodeSampledBitmap(path, 144, 144);
                    if (thumb != null) {
                        thumbCache.put(path, thumb);
                        runOnUiThread(() -> {
                            if (path.equals(iv.getTag())) {
                                iv.setImageBitmap(thumb);
                            }
                        });
                    }
                });
            }

            return convertView;
        }

        void shutdown() {
            executor.shutdownNow();
            thumbCache.evictAll();
        }

        private class ViewHolder {
            CheckBox cbSelect;
            ImageView ivThumb;
            TextView tvFileName;
        }
    }

    /** 按采样率解码图片生成缩略图，避免OOM */
    private static Bitmap decodeSampledBitmap(String path, int reqWidth, int reqHeight) {
        try {
            final BitmapFactory.Options options = new BitmapFactory.Options();
            options.inJustDecodeBounds = true;
            BitmapFactory.decodeFile(path, options);
            options.inSampleSize = calculateInSampleSize(options, reqWidth, reqHeight);
            options.inJustDecodeBounds = false;
            options.inPreferredConfig = Bitmap.Config.RGB_565;
            return BitmapFactory.decodeFile(path, options);
        } catch (Exception e) {
            return null;
        }
    }

    private static int calculateInSampleSize(BitmapFactory.Options options, int reqWidth, int reqHeight) {
        final int height = options.outHeight;
        final int width = options.outWidth;
        int inSampleSize = 1;
        if (height > reqHeight || width > reqWidth) {
            final int halfHeight = height / 2;
            final int halfWidth = width / 2;
            while ((halfHeight / inSampleSize) >= reqHeight
                    && (halfWidth / inSampleSize) >= reqWidth) {
                inSampleSize *= 2;
            }
        }
        return inSampleSize;
    }
}
