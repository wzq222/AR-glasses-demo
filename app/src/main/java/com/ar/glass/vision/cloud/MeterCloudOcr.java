package com.ar.glass.vision.cloud;

import android.graphics.Bitmap;
import android.util.Base64;
import android.util.Log;

import com.ar.glass.BuildConfig;
import com.ar.glass.vision.MeterReading;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

/**
 * 万用表读数云端识别引擎（OpenAI 兼容视觉大模型 API）。
 *
 * 一次调用同时识别：屏幕数字读数 + 单位符号 + 旋钮挡位 + 表笔插孔异常，
 * 返回结构化结果 {@link MeterReading}。
 *
 * 支持任意 OpenAI Chat Completions 兼容的视觉模型服务商，默认通义千问 qwen3-vl-flash：
 * - 通义千问（阿里云百炼兼容接口，已验证图片输入）：
 *   BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
 *   MODEL=qwen3-vl-flash（flash 视觉版，速度快；准确率要求更高可换 qwen3.8-max）
 * - 小米 MiMo：platform.xiaomimimo.com 开通，视觉模型 mimo-v2.5
 *   BASE_URL=https://api.xiaomimimo.com/v1/chat/completions
 * - 火山引擎豆包：方舟(Ark) 开通视觉模型
 *   BASE_URL=https://ark.cn-beijing.volces.com/api/v3/chat/completions
 *
 * 接入方式：
 *   1. 在所选平台开通视觉模型并创建 API Key；
 *   2. 在项目根目录 gradle.properties 填写：
 *        ARK_BASE_URL=接口地址（可选，默认通义千问）
 *        ARK_API_KEY=你的key
 *        ARK_MODEL=模型名（可选，默认 qwen3-vl-flash）
 *   3. 重新编译，BuildConfig 自动注入。
 *
 * 调用流程：图片缩放压缩 → Base64 → POST /chat/completions
 * → 解析 choices[0].message.content 中的 JSON 得到结构化读数。
 *
 * 注意：网络请求为耗时操作，务必在后台线程调用（勿在主线程执行）。
 */
public final class MeterCloudOcr {

    private static final String TAG = "MeterCloudOcr";

    /** 默认接口地址（可用 gradle.properties 的 ARK_BASE_URL 覆盖）。 */
    private static final String DEFAULT_BASE_URL =
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions";

    /** 默认模型（可用 gradle.properties 的 ARK_MODEL 覆盖）。 */
    private static final String DEFAULT_MODEL = "qwen3-vl-flash";

    private static final int CONNECT_TIMEOUT_MS = 15000;
    private static final int READ_TIMEOUT_MS = 30000;
    /** 发送前最长边限制：1024 足够数字识别，兼顾上传速度（实测准确率不变）。 */
    private static final int MAX_EDGE = 1024;
    private static final int JPEG_QUALITY = 75;

    /**
     * 识别提示词：输出结构化 JSON，同时识别读数、单位、挡位、异常。
     */
    private static final String PROMPT =
            "你是一块高精度数字万用表的「读数 + 挡位」识别器。请识别图片中的万用表：\n" +
            "1) 屏幕（LCD/LED 数码显示）上的数字读数与单位符号；\n" +
            "2) 旋钮所在挡位（若照片能看到旋钮/拨盘）；\n" +
            "3) 判断表笔插孔是否与挡位匹配（若照片能看到插孔）。\n" +
            "只输出一个 JSON 对象，不要任何解释，不要 markdown 代码块，格式如下：\n" +
            "{\"reading\":\"24.8\",\"unit\":\"V\",\"gear\":\"直流电压\",\"warning\":\"\"}\n" +
            "字段说明：\n" +
            "- reading：屏幕上的数字读数，逐位仔细核对 8/0/3、1/7、小数点和负号；无读数填 \"\"\n" +
            "- unit：屏幕上显示的单位符号，如 V/mV/A/mA/Ω/kΩ/MΩ；无单位填 \"\"\n" +
            "- gear：旋钮挡位，如\"直流电压\"\"交流电压\"\"电阻\"\"通断蜂鸣\"\"直流电流\"\"交流电流\"\"电容\"\"温度\"；看不到旋钮填 \"\"\n" +
            "- warning：简短的异常提示，如\"表笔插孔与挡位不符\"；无异常填 \"\"\n" +
            "如果画面中没有万用表或读不出任何读数，四个字段都填 \"\"。";

    /** 最近一次调用的错误信息（用于排查）。 */
    private static volatile String sLastError;

    private MeterCloudOcr() {
    }

    /**
     * 识别图片中的万用表读数与挡位。
     *
     * @param bitmap 待识别图片（万用表照片）
     * @return 结构化识别结果；未识别到或调用失败返回 null
     */
    public static MeterReading recognizeMeter(Bitmap bitmap) {
        sLastError = null;
        String apiKey = BuildConfig.ARK_API_KEY;
        if (apiKey == null || apiKey.trim().isEmpty()) {
            sLastError = "未配置 ARK_API_KEY：请在 gradle.properties 中填写所选平台的 API Key 后重新编译";
            Log.w(TAG, sLastError);
            return null;
        }
        if (bitmap == null) {
            sLastError = "输入图片为空";
            return null;
        }

        String base64 = compressToBase64(bitmap);
        if (base64 == null) {
            sLastError = "图片压缩/编码失败";
            return null;
        }

        String baseUrl = BuildConfig.ARK_BASE_URL;
        if (baseUrl == null || baseUrl.trim().isEmpty()) {
            baseUrl = DEFAULT_BASE_URL;
        }
        String model = BuildConfig.ARK_MODEL;
        if (model == null || model.trim().isEmpty()) {
            model = DEFAULT_MODEL;
        }

        try {
            JSONObject body = new JSONObject();
            body.put("model", model);
            body.put("temperature", 0);
            body.put("max_tokens", 200);

            JSONArray content = new JSONArray();
            content.put(new JSONObject().put("type", "text").put("text", PROMPT));
            JSONObject imageUrl = new JSONObject()
                    .put("url", "data:image/jpeg;base64," + base64);
            content.put(new JSONObject()
                    .put("type", "image_url")
                    .put("image_url", imageUrl));

            JSONObject userMsg = new JSONObject()
                    .put("role", "user")
                    .put("content", content);
            body.put("messages", new JSONArray().put(userMsg));

            String resp = postJson(baseUrl, apiKey, body.toString());
            return parseMeterReading(resp);
        } catch (Exception e) {
            sLastError = "请求/解析异常：" + e.getMessage();
            Log.e(TAG, "识别失败", e);
            return null;
        }
    }

    /** 获取最近一次调用的错误信息。 */
    public static String getLastError() {
        return sLastError;
    }

    /** 缩放 + JPEG 压缩 + Base64 编码。 */
    private static String compressToBase64(Bitmap src) {
        Bitmap scaled = src;
        int maxEdge = Math.max(src.getWidth(), src.getHeight());
        if (maxEdge > MAX_EDGE) {
            double ratio = MAX_EDGE / (double) maxEdge;
            int newW = Math.max(1, (int) (src.getWidth() * ratio));
            int newH = Math.max(1, (int) (src.getHeight() * ratio));
            scaled = Bitmap.createScaledBitmap(src, newW, newH, true);
        }
        try {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            scaled.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, baos);
            byte[] bytes = baos.toByteArray();
            if (scaled != src) {
                scaled.recycle();
            }
            if (bytes.length == 0) {
                return null;
            }
            return Base64.encodeToString(bytes, Base64.NO_WRAP);
        } catch (Exception e) {
            Log.e(TAG, "压缩失败", e);
            if (scaled != src) {
                scaled.recycle();
            }
            return null;
        }
    }

    /** POST JSON 到指定地址（Bearer 鉴权）。 */
    private static String postJson(String urlStr, String apiKey, String jsonBody) throws Exception {
        HttpURLConnection conn = null;
        try {
            URL url = new URL(urlStr);
            conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setConnectTimeout(CONNECT_TIMEOUT_MS);
            conn.setReadTimeout(READ_TIMEOUT_MS);
            conn.setDoOutput(true);
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setRequestProperty("Authorization", "Bearer " + apiKey);

            byte[] bodyBytes = jsonBody.getBytes(StandardCharsets.UTF_8);
            conn.setFixedLengthStreamingMode(bodyBytes.length);
            try (OutputStream os = conn.getOutputStream()) {
                os.write(bodyBytes);
                os.flush();
            }

            int code = conn.getResponseCode();
            if (code != HttpURLConnection.HTTP_OK) {
                String err = readStream(conn.getErrorStream());
                sLastError = "HTTP " + code + ": " + err;
                Log.e(TAG, "HTTP " + code + ": " + err);
                return null;
            }
            return readStream(conn.getInputStream());
        } finally {
            if (conn != null) {
                conn.disconnect();
            }
        }
    }

    private static String readStream(InputStream in) throws Exception {
        if (in == null) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(in, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
        }
        return sb.toString();
    }

    /**
     * 解析 OpenAI 兼容响应，提取 choices[0].message.content，
     * 优先解析为结构化 JSON，失败则回退为纯文本读数。
     */
    private static MeterReading parseMeterReading(String resp) {
        if (resp == null || resp.trim().isEmpty()) {
            return null;
        }
        try {
            JSONObject root = new JSONObject(resp);
            JSONArray choices = root.optJSONArray("choices");
            if (choices == null || choices.length() == 0) {
                sLastError = "响应中没有 choices 字段：" + resp;
                return null;
            }
            JSONObject message = choices.getJSONObject(0).optJSONObject("message");
            if (message == null) {
                sLastError = "响应中没有 message 字段";
                return null;
            }
            String content = message.optString("content", "").trim();
            if (content.isEmpty() || "UNKNOWN".equalsIgnoreCase(content)) {
                return null;
            }

            MeterReading r = new MeterReading();
            r.raw = content;

            String json = extractJson(content);
            if (json != null) {
                try {
                    JSONObject o = new JSONObject(json);
                    r.value = o.optString("reading", "").trim();
                    r.unit = o.optString("unit", "").trim();
                    r.gear = o.optString("gear", "").trim();
                    r.warning = o.optString("warning", "").trim();
                } catch (Exception e) {
                    // JSON 解析失败，走纯文本兜底
                    json = null;
                }
            }
            if (json == null) {
                // 兜底：把整段内容当作纯文本读数（兼容旧行为，如 "24.8 V"）
                r.value = content;
            }

            if ("UNKNOWN".equalsIgnoreCase(r.value)) {
                r.value = "";
            }
            // 四个字段全空视为未识别
            if (r.value.isEmpty() && r.unit.isEmpty() && r.gear.isEmpty() && r.warning.isEmpty()) {
                return null;
            }
            return r;
        } catch (Exception e) {
            sLastError = "响应解析失败：" + e.getMessage();
            Log.e(TAG, "解析失败，原始响应: " + resp, e);
            return null;
        }
    }

    /** 从模型输出中提取第一个 JSON 对象（自动去掉 markdown 代码块与多余文字）。 */
    private static String extractJson(String content) {
        if (content == null) {
            return null;
        }
        String s = content.trim();
        s = s.replaceAll("(?s)```json", "").replaceAll("(?s)```", "").trim();
        int start = s.indexOf('{');
        int end = s.lastIndexOf('}');
        if (start >= 0 && end > start) {
            return s.substring(start, end + 1);
        }
        return null;
    }
}
