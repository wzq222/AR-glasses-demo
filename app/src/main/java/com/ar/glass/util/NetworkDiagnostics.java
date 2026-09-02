package com.ar.glass.util;

import android.content.Context;
import android.net.wifi.WifiConfiguration;
import android.net.wifi.WifiInfo;
import android.net.wifi.WifiManager;
import android.net.wifi.p2p.WifiP2pDevice;
import android.os.Build;

import com.ar.glass.core.AppState;
import com.ar.glass.core.GlassBleService;

import java.net.Inet4Address;
import java.net.InetAddress;
import java.util.Collection;
import java.util.Enumeration;

/**
 * 网络诊断报告：收集 WiFi / 热点 / WiFi Direct(P2P) / 网络接口 / ARP 表 的完整状态，
 * 用于排查「眼镜拍照 → P2P/热点回传」链路失败原因。
 *
 * 触发方式：
 *   - APP 内「🌐 网络诊断」按钮（结果写入运行日志）
 *   - adb shell am broadcast -a com.ar.glass.DEBUG --es cmd netdiag
 */
public final class NetworkDiagnostics {

    private NetworkDiagnostics() {}

    /** 收集完整诊断报告（多行文本，直接进运行日志/logcat） */
    public static String collect(Context context) {
        StringBuilder sb = new StringBuilder();
        sb.append("========== 网络诊断报告 ==========\n");

        // 1. WiFi 基础状态
        try {
            WifiManager wm = (WifiManager) context.getApplicationContext()
                    .getSystemService(Context.WIFI_SERVICE);
            if (wm == null) {
                sb.append("[WiFi] 无 WifiManager\n");
            } else {
                sb.append("[WiFi] 开关=").append(wm.isWifiEnabled() ? "开" : "关").append("\n");
                WifiInfo info = wm.getConnectionInfo();
                if (info != null && info.getNetworkId() != -1) {
                    String ssid = info.getSSID().replace("\"", "");
                    sb.append("[WiFi] 当前连接 SSID=").append(ssid)
                            .append(" IP=").append(intToIp(info.getIpAddress()))
                            .append(" 信号=").append(wifiSignal(info.getRssi())).append("\n");
                    android.net.DhcpInfo dhcp = wm.getDhcpInfo();
                    if (dhcp != null) {
                        sb.append("[WiFi] DHCP 网关=").append(intToIp(dhcp.gateway))
                                .append(" 本机=").append(intToIp(dhcp.ipAddress)).append("\n");
                    }
                } else {
                    sb.append("[WiFi] 未连接任何 AP\n");
                }
                // 已保存的含眼镜名的网络
                try {
                    for (WifiConfiguration c : wm.getConfiguredNetworks()) {
                        if (c.SSID != null && (c.SSID.contains("CY") || c.SSID.contains("_"))) {
                            sb.append("[WiFi已保存] ").append(c.SSID)
                                    .append(" status=").append(c.status).append("\n");
                        }
                    }
                } catch (Exception ignored) {}
            }
        } catch (Throwable e) {
            sb.append("[WiFi] 采集失败: ").append(e.getMessage()).append("\n");
        }

        // 2. 热点（AP）状态 —— 定位「热点开了但接口没 IP / 眼镜没连上」
        sb.append("[热点]\n");
        try {
            WifiManager wm = (WifiManager) context.getApplicationContext()
                    .getSystemService(Context.WIFI_SERVICE);
            try {
                Object state = WifiManager.class.getMethod("getWifiApState").invoke(wm);
                sb.append("  状态=").append(apStateName(state)).append("\n");
            } catch (Throwable e) {
                sb.append("  状态=获取失败(系统限制: ").append(e.getClass().getSimpleName()).append(")\n");
            }
            try {
                WifiConfiguration ap = (WifiConfiguration)
                        WifiManager.class.getMethod("getWifiApConfiguration").invoke(wm);
                if (ap != null && ap.SSID != null) {
                    sb.append("  SSID=").append(ap.SSID.replace("\"", ""))
                            .append(" 加密=").append(ap.preSharedKey == null ? "open" : "WPA2").append("\n");
                }
            } catch (Throwable ignored) {}
            try {
                android.net.ConnectivityManager cm = (android.net.ConnectivityManager)
                        context.getApplicationContext().getSystemService(Context.CONNECTIVITY_SERVICE);
                String[] tethered = (String[]) android.net.ConnectivityManager.class
                        .getMethod("getTetheredIfaces").invoke(cm);
                sb.append("  tether接口=").append(java.util.Arrays.toString(tethered)).append("\n");
            } catch (Throwable e) {
                sb.append("  tether接口=获取失败(系统限制: ").append(e.getClass().getSimpleName()).append(")\n");
            }
        } catch (Throwable e) {
            sb.append("  采集失败: ").append(e.getMessage()).append("\n");
        }

        // 3. WiFi Direct (P2P) 状态
        try {
            GlassBleService s = GlassBleService.debugInstance();
            if (s == null) {
                sb.append("[P2P] BLE服务未运行\n");
            } else {
                sb.append(s.p2pDiagnosticSnapshot());
            }
        } catch (Throwable e) {
            sb.append("[P2P] 采集失败: ").append(e.getMessage()).append("\n");
        }

        // 3. 网络接口
        sb.append("[接口]\n");
        try {
            Enumeration<java.net.NetworkInterface> list =
                    java.net.NetworkInterface.getNetworkInterfaces();
            while (list.hasMoreElements()) {
                java.net.NetworkInterface nif = list.nextElement();
                if (!nif.isUp() || nif.isLoopback()) continue;
                StringBuilder ips = new StringBuilder();
                Enumeration<InetAddress> addrs = nif.getInetAddresses();
                while (addrs.hasMoreElements()) {
                    InetAddress a = addrs.nextElement();
                    if (a instanceof Inet4Address) {
                        ips.append(a.getHostAddress()).append(" ");
                    }
                }
                if (ips.length() > 0) {
                    sb.append("  ").append(nif.getName()).append(" => ").append(ips.toString().trim()).append("\n");
                }
            }
        } catch (Throwable e) {
            sb.append("  采集失败: ").append(e.getMessage()).append("\n");
        }

        // 5. 邻居表（ip neigh 为 Android 10+ 正途，ARP 表兜底；热点模式下眼镜接入后会出现在这里）
        sb.append("[邻居表]\n");
        java.util.Set<String> neigh = readNeighbors();
        if (neigh.isEmpty()) {
            sb.append("  （空）\n");
        } else {
            for (String n : neigh) sb.append("  ").append(n).append("\n");
        }

        // 6. 应用层回传状态 + 主动探测
        AppState st = AppState.getInstance();
        sb.append("[回传状态] BLE=").append(st.isBleConnected)
                .append(" 眼镜IP=").append(st.serverIp)
                .append(" socket=").append(st.isSocketConnected).append("\n");
        sb.append("[主动探测]\n");
        try {
            String kip = com.xy.ksdk.api.wifi.WifiConnector.getInstance().getServerIP();
            if (kip != null && !kip.isEmpty()) sb.append("  KSDK报告IP=").append(kip).append("\n");
        } catch (Throwable ignored) {}
        if (st.serverIp == null || st.serverIp.isEmpty()) {
            sb.append("  （眼镜IP未获取，跳过。可等 BLE 上报或用 setip 命令注入）\n");
        } else {
            String ip = st.serverIp;
            try {
                boolean ok = InetAddress.getByName(ip).isReachable(1500);
                sb.append("  ping ").append(ip).append(" => ").append(ok ? "可达" : "不可达(ICMP可能被禁)").append("\n");
            } catch (Throwable e) {
                sb.append("  ping ").append(ip).append(" => 异常 ").append(e.getClass().getSimpleName()).append("\n");
            }
            for (String path : new String[]{"/files/media.config", "/storage/sd0/C/DCIM/1/vf_list.txt"}) {
                sb.append("  GET http://").append(ip).append(path).append(" => ")
                        .append(httpProbe(ip, path)).append("\n");
            }
        }

        sb.append("==================================");
        return sb.toString();
    }

    /** 读取邻居表：优先 ip neigh，失败退回 /proc/net/arp */
    public static java.util.Set<String> readNeighbors() {
        java.util.Set<String> out = new java.util.LinkedHashSet<>();
        try {
            Process p = Runtime.getRuntime().exec(new String[]{"sh", "-c", "ip neigh show"});
            java.io.BufferedReader r = new java.io.BufferedReader(
                    new java.io.InputStreamReader(p.getInputStream()));
            String line;
            while ((line = r.readLine()) != null) {
                line = line.trim();
                if (!line.isEmpty()) out.add(line);
            }
            r.close();
            p.destroy();
        } catch (Throwable ignored) {}
        if (out.isEmpty()) {
            try {
                java.io.BufferedReader r = new java.io.BufferedReader(
                        new java.io.FileReader("/proc/net/arp"));
                String line;
                int n = 0;
                while ((line = r.readLine()) != null) {
                    line = line.trim();
                    if (n++ > 0 && !line.isEmpty()) out.add("arp: " + line);
                }
                r.close();
            } catch (Throwable ignored) {}
        }
        return out;
    }

    /** 单路径 HTTP 探测，返回状态码或具体失败原因 */
    private static String httpProbe(String ip, String path) {
        long t0 = android.os.SystemClock.elapsedRealtime();
        try {
            java.net.HttpURLConnection conn = (java.net.HttpURLConnection)
                    new java.net.URL("http://" + ip + path).openConnection();
            conn.setConnectTimeout(1500);
            conn.setReadTimeout(1500);
            int code = conn.getResponseCode();
            conn.disconnect();
            long ms = android.os.SystemClock.elapsedRealtime() - t0;
            return code == 200 ? "✅ 200 (" + ms + "ms)" : "HTTP " + code + " (" + ms + "ms)";
        } catch (Throwable e) {
            long ms = android.os.SystemClock.elapsedRealtime() - t0;
            String reason = e instanceof java.net.SocketTimeoutException ? "连接超时"
                    : e instanceof java.net.ConnectException ? "连接被拒绝"
                    : e instanceof java.net.NoRouteToHostException
                      || e instanceof java.net.UnknownHostException ? "不可达/无法解析"
                    : e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName();
            return "❌ " + reason + " (" + ms + "ms)";
        }
    }

    private static String apStateName(Object raw) {
        int s;
        try { s = raw instanceof Number ? ((Number) raw).intValue() : Integer.parseInt(String.valueOf(raw)); }
        catch (Exception e) { return String.valueOf(raw); }
        switch (s) {
            case 10: return "关闭中(10)";
            case 11: return "已关闭(11)";
            case 12: return "开启中(12)";
            case 13: return "已开启(13) ✅";
            case 14: return "失败(14)";
            default: return "未知(" + s + ")";
        }
    }

    private static String intToIp(int ip) {
        if (ip == 0) return "0.0.0.0";
        return (ip & 0xFF) + "." + ((ip >> 8) & 0xFF) + "." + ((ip >> 16) & 0xFF)
                + "." + ((ip >> 24) & 0xFF);
    }

    private static String wifiSignal(int rssi) {
        if (rssi >= -50) return "极强";
        if (rssi >= -65) return "强";
        if (rssi >= -75) return "中";
        return "弱(" + rssi + "dBm)";
    }
}
