package com.ar.glass.util;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.net.wifi.WifiManager;
import android.util.Log;

import com.ar.glass.core.AppState;
import com.ar.glass.util.EventMsg;
import com.xy.ksdk.api.cmd.CmdSendManager;
import com.xy.ksdk.cmd.base.SCmd;

import org.greenrobot.eventbus.EventBus;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.Inet4Address;
import java.net.URL;
import java.util.Enumeration;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * ADB 调试控制接口：批量测试/验证回传链路各环节。
 *
 * 用法（cmd 必填，其余按命令选填）：
 *   adb shell am broadcast -a com.ar.glass.DEBUG --es cmd status
 *   adb shell am broadcast -a com.ar.glass.DEBUG --es cmd interfaces
 *   adb shell am broadcast -a com.ar.glass.DEBUG --es cmd hotspot          # 开热点并返回 SSID/密码
 *   adb shell am broadcast -a com.ar.glass.DEBUG --es cmd stop_hotspot
 *   adb shell am broadcast -a com.ar.glass.DEBUG --es cmd setap --es ssid SS --es pwd PP
 *                                                          # 保存外部AP配置（最稳定路径：手机与眼镜同连路由器）
 *   adb shell am broadcast -a com.ar.glass.DEBUG --es cmd ap            # 查看外部AP配置
 *   adb shell am broadcast -a com.ar.glass.DEBUG --es cmd sta --es ssid SS --es pwd PP
 *                                                          # 向眼镜发 cs_wfsta 配网命令
 *   adb shell am broadcast -a com.ar.glass.DEBUG --es cmd raw --es c cs_xxx --es k1 v1 --es k2 v2
 *                                                          # 透传任意 ksdk 命令
 *   adb shell am broadcast -a com.ar.glass.DEBUG --es cmd scan             # 扫描本机所有网段找眼镜 HTTP
 *   adb shell am broadcast -a com.ar.glass.DEBUG --es cmd probe --es ip 192.168.43.100
 *   adb shell am broadcast -a com.ar.glass.DEBUG --es cmd photo            # BLE 拍照
 *   adb shell am broadcast -a com.ar.glass.DEBUG --es cmd detect           # 检测本地最新照片
 *   adb shell am broadcast -a com.ar.glass.DEBUG --es cmd photo_sync       # 拍照→4s→强制同步（原单张流程）
 *   adb shell am broadcast -a com.ar.glass.DEBUG --es cmd gap              # 眼镜AP模式①：BLE请求眼镜开热点(cs_opap)
 *   adb shell am broadcast -a com.ar.glass.DEBUG --es cmd gapconn --es ssid SS --es pwd PP
 *                                                          # 眼镜AP模式②：手机连接眼镜AP（开放网络 pwd 传 open）
 *   adb shell am broadcast -a com.ar.glass.DEBUG --es cmd gapclose         # 眼镜AP模式收尾：关AP+断连
 */
public class GlassDebugReceiver extends BroadcastReceiver {

    private static final String TAG = "GlassDebug";

    @Override
    public void onReceive(Context context, Intent intent) {
        String cmd = intent.getStringExtra("cmd");
        log("收到调试命令: " + cmd);
        if (cmd == null) return;
        final Context app = context.getApplicationContext();
        switch (cmd) {
            case "status": {
                log("BLE连接=" + AppState.getInstance().isBleConnected
                        + " 系统就绪=" + AppState.getInstance().isSystemReady
                        + " 眼镜IP=" + AppState.getInstance().serverIp
                        + " 电量=" + AppState.getInstance().batteryLevel + "%");
                log("网络接口: " + listInterfaces());
                break;
            }
            case "interfaces": {
                log("网络接口: " + listInterfaces());
                break;
            }
            case "hotspot": {
                new Thread(() -> {
                    WifiManager wm = (WifiManager) app.getSystemService(Context.WIFI_SERVICE);
                    if (wm == null) { log("无 WifiManager"); return; }
                    wm.startLocalOnlyHotspot(new WifiManager.LocalOnlyHotspotCallback() {
                        @Override
                        public void onStarted(WifiManager.LocalOnlyHotspotReservation r) {
                            try {
                                WifiConfigurationPlaceholder.keep();
                                android.net.wifi.WifiConfiguration c = r.getWifiConfiguration();
                                log("热点开启 SSID=" + (c != null ? c.SSID : "?")
                                        + " PWD=" + (c != null ? c.preSharedKey : "?"));
                            } catch (Throwable e) {
                                log("热点凭证读取失败: " + e.getMessage());
                            }
                        }
                        @Override
                        public void onFailed(int reason) { log("热点失败 reason=" + reason); }
                    }, new android.os.Handler(android.os.Looper.getMainLooper()));
                }).start();
                break;
            }
            case "stop_hotspot": {
                GlassBleServiceBridge.stopHotspot();
                log("已请求关闭热点");
                break;
            }
            case "setap": {
                // 保存外部 AP 配置（手机与眼镜同连一台路由器）：--es ssid SS --es pwd PP
                String ssid = intent.getStringExtra("ssid");
                String pwd = intent.getStringExtra("pwd");
                if (ssid == null || pwd == null) { log("缺少 --es ssid / --es pwd"); break; }
                app.getSharedPreferences("debug", Context.MODE_PRIVATE).edit()
                        .putString("ext_ap_ssid", ssid).putString("ext_ap_pwd", pwd).apply();
                log("外部AP已保存 ssid=" + ssid + "（手机连接此 WiFi 后单张检测将自动走外部AP模式）");
                break;
            }
            case "ap": {
                String ssid = app.getSharedPreferences("debug", Context.MODE_PRIVATE)
                        .getString("ext_ap_ssid", "");
                String pwd = app.getSharedPreferences("debug", Context.MODE_PRIVATE)
                        .getString("ext_ap_pwd", "");
                log("外部AP配置: ssid=" + ssid + " pwd=" + (pwd == null || pwd.isEmpty() ? "未设置" : "已设置"));
                break;
            }
            case "sta": {
                String ssid = intent.getStringExtra("ssid");
                String pwd = intent.getStringExtra("pwd");
                if (ssid == null || pwd == null) { log("缺少 --es ssid / --es pwd"); break; }
                initCmdSender(app);
                new Thread(() -> CmdSendManager.getInstance()
                        .sendWifiSTA(ssid, pwd)).start();
                log("已发送 cs_wfsta ssid=" + ssid + " pwd=" + pwd + "（5 秒后自动重发一次）");
                new Thread(() -> {
                    try { Thread.sleep(5000); } catch (Exception ignored) {}
                    CmdSendManager.getInstance().sendWifiSTA(ssid, pwd);
                }).start();
                break;
            }
            case "raw": {
                String c = intent.getStringExtra("c");
                if (c == null) { log("缺少 --es c"); break; }
                initCmdSender(app);
                new Thread(() -> {
                    try {
                        RawKsCmd raw = new RawKsCmd(c);
                        java.util.Set<String> keys = intent.getExtras() != null
                                ? intent.getExtras().keySet() : new java.util.HashSet<>();
                        for (String k : keys) {
                            if (k.equals("cmd") || k.equals("c")) continue;
                            raw.kvString(k, intent.getStringExtra(k));
                        }
                        log("raw 发送: " + c);
                        raw.sendBridge();
                    } catch (Throwable e) {
                        log("raw 发送异常: " + e.getMessage());
                    }
                }).start();
                break;
            }
            case "scan": {
                new Thread(() -> {
                    Set<String> hits = scanAllSubnets(app);
                    log("扫描结果: " + (hits.isEmpty() ? "未发现眼镜 HTTP 服务" : hits));
                }).start();
                break;
            }
            case "probe": {
                String ip = intent.getStringExtra("ip");
                if (ip == null) { log("缺少 --es ip"); break; }
                new Thread(() -> log("probe " + ip + " -> " + probe(ip))).start();
                break;
            }
            case "photo": {
                new Thread(() -> {
                    GlassBleServiceBridge.takePhoto();
                    log("拍照命令已发送");
                }).start();
                break;
            }
            case "detect": {
                new DetectSelfTestReceiver().sendDetect(app);
                break;
            }
            case "photo_sync": {
                new Thread(() -> {
                    GlassBleServiceBridge.takePhoto();
                    try { Thread.sleep(4000); } catch (Exception ignored) {}
                    log("photo_sync: 触发同步");
                    GlassBleServiceBridge.startSync();
                }).start();
                break;
            }
            case "bleget": {
                // 拍照 → 4s → BLE 直传拉取照片（无需 WiFi，最稳定路径）
                GlassBleServiceBridge.bleGet();
                log("bleget: 拍照后 4s 经 BLE 请求推送照片");
                break;
            }
            case "blesdfl": {
                GlassBleServiceBridge.bleList();
                log("blesdfl: 已请求眼镜文件列表");
                break;
            }
            case "bleprogress": {
                log("BLE接收状态: " + GlassBleServiceBridge.bleProgress());
                break;
            }
            case "gap": {
                // 眼镜 AP 模式第一步：BLE 请求眼镜开启热点（cs_opap）
                initCmdSender(app);
                new Thread(() -> {
                    com.xy.ksdk.api.cmd.CmdSendManager.getInstance().requestServerOpenHotspot();
                    log("已请求眼镜开启 AP（cs_opap），5s 后自动重发一次；回报见 [BLE命令] 日志");
                }).start();
                new Thread(() -> {
                    try { Thread.sleep(5000); } catch (Exception ignored) {}
                    com.xy.ksdk.api.cmd.CmdSendManager.getInstance().requestServerOpenHotspot();
                    log("cs_opap 重发完成");
                }).start();
                break;
            }
            case "gapconn": {
                // 眼镜 AP 模式第二步：手机连接眼镜 AP（--es ssid SS --es pwd PP，开放网络 pwd 传 open）
                String ssid = intent.getStringExtra("ssid");
                String pwd = intent.getStringExtra("pwd");
                if (ssid == null) { log("缺少 --es ssid（眼镜AP名通常为 BLE名_MAC，可从状态日志获取）"); break; }
                String password = "open".equals(pwd) || pwd == null ? "" : pwd;
                new Thread(() -> {
                    com.xy.ksdk.api.wifi.WifiConnector.getInstance().init(app,
                            new com.xy.ksdk.api.wifi.WifiConnector.WifiListener() {
                                @Override
                                public void onWifiConnectResult(long code, boolean ok) {
                                    log("WiFi连接回调: ok=" + ok + " code=" + code
                                            + " 当前SSID=" + com.xy.ksdk.api.wifi.WifiConnector.getInstance().getCurrentWifiSsid());
                                    if (ok) {
                                        String ip = com.xy.ksdk.api.wifi.WifiConnector.getInstance().getServerIP();
                                        log("眼镜AP已连上，网关IP=" + ip + "，可执行 cmd scan 探测眼镜HTTP服务");
                                    }
                                }

                                @Override
                                public void onWifiConnectTimeout() {
                                    log("WiFi连接超时（检查AP名/密码）");
                                }
                            });
                    boolean r = com.xy.ksdk.api.wifi.WifiConnector.getInstance().connect(ssid, password);
                    log("connect(" + ssid + ") 已发起: " + r);
                }).start();
                break;
            }
            case "gapclose": {
                // 请求眼镜关闭 AP
                initCmdSender(app);
                new Thread(() -> {
                    com.xy.ksdk.api.cmd.CmdSendManager.getInstance().closeHotspot();
                    com.xy.ksdk.api.wifi.WifiConnector.getInstance().disconnect();
                    log("已请求眼镜关闭 AP（cs_csap）并断开手机连接");
                }).start();
                break;
            }
            case "setgapwd": {
                // 设置眼镜 AP 密码（眼镜AP为加密网络时用）：--es pwd PP
                String pwd = intent.getStringExtra("pwd");
                if (pwd == null) { log("缺少 --es pwd"); break; }
                app.getSharedPreferences("debug", Context.MODE_PRIVATE)
                        .edit().putString("gap_pwd", pwd).apply();
                log("眼镜AP密码已保存（自动流程连接加密AP时使用）");
                break;
            }
            default:
                log("未知命令: " + cmd);
        }
    }

    // ---- 工具 ----

    private static void log(String text) {
        Log.i(TAG, text);
        EventBus.getDefault().post(new EventMsg(EventMsg.MSG_LOG, "🛠 " + text));
    }

    private static String listInterfaces() {
        StringBuilder sb = new StringBuilder();
        try {
            Enumeration<java.net.NetworkInterface> list =
                    java.net.NetworkInterface.getNetworkInterfaces();
            while (list.hasMoreElements()) {
                java.net.NetworkInterface nif = list.nextElement();
                if (!nif.isUp() || nif.isLoopback()) continue;
                Enumeration<java.net.InetAddress> addrs = nif.getInetAddresses();
                while (addrs.hasMoreElements()) {
                    java.net.InetAddress a = addrs.nextElement();
                    if (a instanceof Inet4Address) {
                        sb.append(nif.getName()).append("=").append(a.getHostAddress()).append(" ");
                    }
                }
            }
        } catch (Exception e) {
            sb.append("error ").append(e.getMessage());
        }
        return sb.toString().trim();
    }

    private static void initCmdSender(Context app) {
        CmdSendManager m = CmdSendManager.getInstance();
        m.init(app);
        m.setBluetooth(new com.xy.ksdk.api.cmd.IBluetooth() {
            @Override
            public void write(byte[] data) {
                Log.i(TAG, "📤 [KSDK] " + new String(data));
                GlassBleServiceBridge.writeRaw(data);
            }
        });
    }

    /** 透传任意 ksdk 命令（继承 SCmd 以访问 protected addBodyKV） */
    private static class RawKsCmd extends SCmd {
        RawKsCmd(String cmd) { super(cmd); }
        void kvString(String k, String v) { addBodyKV(k, v); }
        void sendBridge() {
            send(new com.xy.ksdk.api.cmd.IBluetooth() {
                @Override
                public void write(byte[] data) {
                    Log.i(TAG, "📤 [KSDK] " + new String(data));
                    GlassBleServiceBridge.writeRaw(data);
                }
            });
        }
    }

    /** 扫描本机所有网段找眼镜 HTTP（并发 32/网段） */
    private static Set<String> scanAllSubnets(Context app) {
        Set<String> found = java.util.Collections.synchronizedSet(new java.util.LinkedHashSet<>());
        Set<String> selfIps = new java.util.HashSet<>();
        Set<String> subnets = new java.util.LinkedHashSet<>();
        try {
            Enumeration<java.net.NetworkInterface> list =
                    java.net.NetworkInterface.getNetworkInterfaces();
            while (list.hasMoreElements()) {
                java.net.NetworkInterface nif = list.nextElement();
                if (!nif.isUp() || nif.isLoopback()) continue;
                Enumeration<java.net.InetAddress> addrs = nif.getInetAddresses();
                while (addrs.hasMoreElements()) {
                    java.net.InetAddress a = addrs.nextElement();
                    if (a instanceof Inet4Address && a.isSiteLocalAddress()) {
                        String ip = a.getHostAddress();
                        selfIps.add(ip);
                        subnets.add(ip.substring(0, ip.lastIndexOf('.') + 1));
                    }
                }
            }
        } catch (Exception ignored) {}
        log("扫描网段: " + subnets + " 本机: " + selfIps);
        java.util.concurrent.ExecutorService pool =
                java.util.concurrent.Executors.newFixedThreadPool(32);
        java.util.List<java.util.concurrent.Future<?>> futures = new java.util.ArrayList<>();
        for (String base : subnets) {
            for (int i = 1; i <= 254; i++) {
                final String ip = base + i;
                if (selfIps.contains(ip)) continue;
                futures.add(pool.submit(() -> {
                    String r = probe(ip);
                    if (r != null) found.add(ip + " (" + r + ")");
                }));
            }
        }
        for (java.util.concurrent.Future<?> f : futures) {
            try { f.get(3, java.util.concurrent.TimeUnit.SECONDS); } catch (Exception ignored) {}
        }
        pool.shutdown();
        return found;
    }

    /** 探测眼镜 HTTP，返回命中的 URL 路径（media.config 或 vf_list.txt），未命中 null */
    private static String probe(String ip) {
        String[][] paths = {
                {"http://" + ip + "/files/media.config", "/files/media.config"},
                {"http://" + ip + ":80/storage/sd0/C/DCIM/1/vf_list.txt", "/storage/sd0/.../vf_list.txt"},
        };
        for (String[] p : paths) {
            try {
                HttpURLConnection conn = (HttpURLConnection) new URL(p[0]).openConnection();
                conn.setConnectTimeout(600);
                conn.setReadTimeout(600);
                conn.setRequestMethod("GET");
                int code = conn.getResponseCode();
                conn.disconnect();
                if (code == 200) return p[1];
            } catch (Exception ignored) {}
        }
        return null;
    }

    /** 占位：防止 getWifiConfiguration 在部分设备抛错的兜底 */
    private static class WifiConfigurationPlaceholder {
        static void keep() {}
    }
}
