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
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.ar.glass.R;
import com.ar.glass.core.AppState;
import com.ar.glass.core.GlassBleService;
import com.ar.glass.util.EventMsg;
import com.ar.glass.voice.VoiceController;

import org.greenrobot.eventbus.EventBus;
import org.greenrobot.eventbus.Subscribe;
import org.greenrobot.eventbus.ThreadMode;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private static final int PERMISSION_REQUEST_CODE = 100;

    private TextView tvBleStatus;
    private TextView tvSystemStatus;
    private TextView tvDeviceName;
    private TextView tvBatteryStatus;
    private TextView tvLog;

    private Button btnSyncPhotos;
    private Button btnGalleryOriginal;
    private Button btnSelectDevice;
    private Button btnVoice;

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

        appendLog("AR眼镜控制应用启动...");
    }

    private void initViews() {
        tvBleStatus = findViewById(R.id.tvBleStatus);
        tvSystemStatus = findViewById(R.id.tvSystemStatus);
        tvDeviceName = findViewById(R.id.tvDeviceName);
        tvBatteryStatus = findViewById(R.id.tvBatteryStatus);
        tvLog = findViewById(R.id.tvLog);

        btnSyncPhotos = findViewById(R.id.btnSyncFiles);
        btnGalleryOriginal = findViewById(R.id.btnGalleryOriginal);
        btnSelectDevice = findViewById(R.id.btnSelectDevice);
        btnVoice = findViewById(R.id.btnVoice);

        btnSyncPhotos.setOnClickListener(v -> syncPhotos());
        btnGalleryOriginal.setOnClickListener(v -> openGallery(GalleryActivity.MODE_ORIGINAL));
        btnSelectDevice.setOnClickListener(v -> showDeviceDialog());
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

        setControlsEnabled(false);
    }

    private void initVoiceController() {
        mVoiceController = new VoiceController(this, new VoiceController.Listener() {
            @Override
            public void onKeywordDetected(String keyword) {
                appendLog("🎤 识别到关键词: " + keyword);
                if (mBleService != null) {
                    mBleService.takePhoto();
                } else {
                    appendLog("⚠️ BLE服务未就绪，无法拍照");
                }
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

            case EventMsg.MSG_BATTERY_UPDATE:
                int battery = msg.arg1;
                boolean charging = msg.arg2 == 1;
                tvBatteryStatus.setText(battery + "%" + (charging ? " ⚡充电中" : ""));
                tvBatteryStatus.setTextColor(getColor(battery < 20
                        ? android.R.color.holo_red_dark : android.R.color.holo_green_dark));
                break;
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
        logBuilder.append(text).append("\n");
        if (tvLog != null) {
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
        if (mVoiceController != null) {
            mVoiceController.release();
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
