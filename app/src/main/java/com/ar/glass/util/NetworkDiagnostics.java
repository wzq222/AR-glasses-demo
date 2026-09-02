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

        // 2. WiFi Direct (P2P) 状态
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

        // 4. ARP 表（热点模式下眼镜接入后会出现在这里）
        sb.append("[ARP表]\n");
        try {
            java.io.File arp = new java.io.File("/proc/net/arp");
            if (arp.exists()) {
                java.io.BufferedReader r = new java.io.BufferedReader(
                        new java.io.FileReader(arp));
                String line;
                int n = 0;
                while ((line = r.readLine()) != null) {
                    if (n++ > 0 && !line.trim().isEmpty()) sb.append("  ").append(line.trim()).append("\n");
                }
                r.close();
                if (n <= 1) sb.append("  （空）\n");
            } else {
                sb.append("  不可读\n");
            }
        } catch (Throwable e) {
            sb.append("  读取失败: ").append(e.getMessage()).append("\n");
        }

        // 5. 应用层回传状态
        AppState st = AppState.getInstance();
        sb.append("[回传状态] BLE=").append(st.isBleConnected)
                .append(" 眼镜IP=").append(st.serverIp)
                .append(" socket=").append(st.isSocketConnected).append("\n");

        sb.append("==================================");
        return sb.toString();
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
