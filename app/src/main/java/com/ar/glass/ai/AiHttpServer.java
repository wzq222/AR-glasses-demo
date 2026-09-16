package com.ar.glass.ai;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.UUID;

import fi.iki.elonen.NanoHTTPD;

/**
 * 本地 OpenAI 兼容服务端：端口 + API_KEY 鉴权，云端/本地可无缝切换。
 *
 *   POST /v1/chat/completions   （Authorization: Bearer <key> 或 X-API-Key）
 *   GET  /v1/models
 *
 * stream=true 时为**真流式 SSE**：JNI 逐 token 回调 → 管道流实时下发（不是事后回放）。
 * 多轮：按请求里的 messages 拼接最近若干轮历史（含 assistant），保留对话上下文。
 */
public class AiHttpServer extends NanoHTTPD {
    private static final String TAG = "AiHttp";
    private static final String PREFS = "ai";

    /** 拼接进 prompt 的最大历史消息条数（user+assistant，控制 prompt 长度）。 */
    private static final int MAX_HISTORY_MSGS = 8;
    /** 单次生成看门狗超时（毫秒）：超时强制取消，避免长任务占住唯一 session。 */
    private static final long GEN_TIMEOUT_MS = 180_000;

    /** 对外暴露的模型 ID（当前实际加载 MiniCPM-V 4.6，文本+视觉同一模型）。 */
    public static final String MODEL_ID = "minicpm-v4.6-local";

    /** 默认 API Key：本地服务，尽量简单（保留 sk- 前缀以兼容 OpenAI SDK）。 */
    public static final String DEFAULT_API_KEY = "sk-local";

    /** 旧版本自动生成的长 key（sk-local-<16 位十六进制>），读取时自动迁移为简单 key。 */
    private static final java.util.regex.Pattern LEGACY_KEY =
            java.util.regex.Pattern.compile("^sk-local-[0-9a-f]{16}$");

    private final LlmEngine llm;
    private final AiPipeline pipeline;
    private volatile boolean ready = false;

    public AiHttpServer(Context ctx, int port, LlmEngine llm, AiPipeline pipeline) {
        super(port);
        this.llm = llm;
        this.pipeline = pipeline;
    }

    /**
     * 读取 API Key；缺失或仍是旧的长随机 key 时，落为简单 key {@link #DEFAULT_API_KEY}。
     * （本地服务，无需强口令；如需自定义，可直接改 SharedPreferences 的 "api_key"）
     */
    public static String getOrCreateApiKey(Context ctx) {
        SharedPreferences sp = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        String key = sp.getString("api_key", null);
        if (key == null || LEGACY_KEY.matcher(key).matches()) {
            key = DEFAULT_API_KEY;
            sp.edit().putString("api_key", key).apply();
        }
        return key;
    }

    public void markReady() {
        ready = true;
    }

    public boolean checkAuth(Map<String, String> headers, String expectedKey) {
        String auth = headers.get("authorization");
        if (auth != null && auth.replace("Bearer ", "").trim().equals(expectedKey)) {
            return true;
        }
        String xKey = headers.get("x-api-key");
        return xKey != null && xKey.trim().equals(expectedKey);
    }

    @Override
    public Response serve(IHTTPSession session) {
        String uri = session.getUri();
        Map<String, String> headers = session.getHeaders();
        String key = getApiKeyExternal();

        try {
            if ("/v1/models".equals(uri) && Method.GET.equals(session.getMethod())) {
                if (!checkAuth(headers, key)) return json(401, err("invalid api key"));
                JSONObject body = new JSONObject()
                        .put("object", "list")
                        .put("data", new JSONArray()
                                .put(new JSONObject().put("id", MODEL_ID)
                                        .put("object", "model")
                                        .put("owned_by", "local")));
                return json(200, body.toString());
            }

            if ("/v1/chat/completions".equals(uri) && Method.POST.equals(session.getMethod())) {
                if (!checkAuth(headers, key)) return json(401, err("invalid api key"));
                if (!ready || !llm.isLoaded()) return json(503, err("model not loaded yet"));

                String raw = readJsonBody(session);
                JSONObject req = new JSONObject(raw == null ? "{}" : raw);
                boolean stream = req.optBoolean("stream", false);
                java.util.List<String[]> history = parseHistory(req);
                String userText = lastUserText(history);
                if (userText == null) return json(400, err("no user message"));
                if (userText.isEmpty()) return json(400, err("empty message"));
                AiDebug.i("api: body " + (raw == null ? 0 : raw.length()) + " chars, userText head="
                        + (userText.length() > 24 ? userText.substring(0, 24) : userText));

                String id = "chatcmpl-local-" + UUID.randomUUID();
                if (stream) {
                    return sse(id, history); // 真流式：逐 token 下发
                }

                long t0 = System.currentTimeMillis();
                Thread wd = watchdog(GEN_TIMEOUT_MS);
                String reply;
                try {
                    reply = llm.generateChat(history, 1024, 0.7f);
                } finally {
                    wd.interrupt();
                }
                AiDebug.i(String.format("api: history=[%s] %d chars -> %d chars in %d ms",
                        describeHistory(history), userText.length(), reply.length(),
                        System.currentTimeMillis() - t0));
                JSONObject resp = new JSONObject()
                        .put("id", id)
                        .put("object", "chat.completion")
                        .put("model", MODEL_ID)
                        .put("choices", new JSONArray().put(new JSONObject()
                                .put("index", 0)
                                .put("message", new JSONObject()
                                        .put("role", "assistant")
                                        .put("content", reply))
                                .put("finish_reason", "stop")))
                        .put("usage", new JSONObject()
                                .put("prompt_tokens", AiPipeline.estimateTokens(userText))
                                .put("completion_tokens", AiPipeline.estimateTokens(reply)));
                return json(200, resp.toString());
            }

            return json(404, err("not found"));
        } catch (Exception e) {
            AiDebug.e("api error: " + e);
            return json(500, err(e.getMessage()));
        }
    }

    private String getApiKeyExternal() {
        return apiKeyProvider != null ? apiKeyProvider.get() : "";
    }

    /** 由宿主注入 API key 读取器（避免 NanoHTTPD 持有 Context）。 */
    public interface KeyProvider {
        String get();
    }

    private volatile KeyProvider apiKeyProvider;

    public void setKeyProvider(KeyProvider provider) {
        this.apiKeyProvider = provider;
    }

    /**
     * 读取请求体并按 UTF-8 解码。
     *
     * <p><b>必须显式 UTF-8</b>：NanoHTTPD 在 Content-Type 未带 charset 时可能按 ISO-8859-1 解析，
     * 中文会变乱码 —— 模型会当成"一串符号"而答非所问（2026-09-16 实测踩坑）。
     * 兜底：原始读取失败时退回 parseBody，并仅在字符串**看起来是乱码**时才做
     * ISO-8859-1→UTF-8 复原（避免把正常 UTF-8 二次解码弄坏）。
     */
    private static String readJsonBody(IHTTPSession session) {
        try {
            Map<String, String> headers = session.getHeaders();
            String lenStr = headers.get("content-length");
            int len = (lenStr == null) ? -1 : Integer.parseInt(lenStr.trim());
            if (len > 0) {
                byte[] buf = new byte[len];
                java.io.InputStream in = session.getInputStream();
                int off = 0;
                while (off < len) {
                    int n = in.read(buf, off, len - off);
                    if (n <= 0) break;
                    off += n;
                }
                if (off > 0) {
                    return new String(buf, 0, off, StandardCharsets.UTF_8);
                }
            }
        } catch (Exception e) {
            AiDebug.e("read body failed: " + e);
        }
        try {
            Map<String, String> files = new java.util.HashMap<>();
            session.parseBody(files);
            return fixMojibakeIfNeeded(files.get("postData"));
        } catch (Exception e) {
            return null;
        }
    }

    /** 仅在字符串主要为 0x80–0xFF 单字节字符（典型 ISO-8859-1 乱码）时还原为 UTF-8。 */
    private static String fixMojibakeIfNeeded(String s) {
        if (s == null || s.isEmpty()) return s;
        int suspicious = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c >= 0x80 && c <= 0xFF) suspicious++;
        }
        if (suspicious >= 2 && suspicious * 2 > s.length()) {
            return new String(s.getBytes(StandardCharsets.ISO_8859_1),
                    StandardCharsets.UTF_8);
        }
        return s;
    }

    /**
     * 解析 OpenAI messages 为 {role, content} 历史列表，只保留最近 {@link #MAX_HISTORY_MSGS} 条。
     * 无 messages 或全空内容时返回空列表（调用方据此返回 400）。
     */
    private static java.util.List<String[]> parseHistory(JSONObject req) {
        java.util.List<String[]> hist = new java.util.ArrayList<>();
        JSONArray msgs = req.optJSONArray("messages");
        if (msgs == null) return hist;
        for (int i = 0; i < msgs.length(); i++) {
            JSONObject m = msgs.optJSONObject(i);
            if (m == null) continue;
            String role = m.optString("role", "user");
            String content = m.optString("content", "");
            if (content == null) content = "";
            content = content.trim();
            if (content.isEmpty()) continue;
            hist.add(new String[]{role, content});
        }
        if (hist.size() > MAX_HISTORY_MSGS) {
            hist = new java.util.ArrayList<>(
                    hist.subList(hist.size() - MAX_HISTORY_MSGS, hist.size()));
        }
        return hist;
    }

    /** 历史摘要（`role:字符数 …`），用于日志核对多轮上下文是否真的带上。 */
    private static String describeHistory(java.util.List<String[]> hist) {
        StringBuilder sb = new StringBuilder();
        for (String[] m : hist) {
            if (sb.length() > 0) sb.append(' ');
            sb.append(m[0]).append(':').append(m[1] == null ? 0 : m[1].length());
        }
        return sb.toString();
    }

    /** 最后一条 user 消息内容；无 user 消息时退化为最后一条。空历史返回 null。 */
    private static String lastUserText(java.util.List<String[]> hist) {
        if (hist == null || hist.isEmpty()) return null;
        for (int i = hist.size() - 1; i >= 0; i--) {
            if ("user".equals(hist.get(i)[0])) return hist.get(i)[1];
        }
        return hist.get(hist.size() - 1)[1];
    }

    /** 生成看门狗：超时后强制取消当前生成。调用方正常结束时 interrupt() 即可。 */
    private Thread watchdog(long timeoutMs) {
        Thread t = new Thread(() -> {
            try {
                Thread.sleep(timeoutMs);
            } catch (InterruptedException e) {
                return; // 已正常结束
            }
            AiDebug.e("gen timeout " + timeoutMs + "ms -> cancel generation");
            try {
                llm.cancel();
            } catch (Throwable ignored) {
            }
        }, "ai-gen-watchdog");
        t.setDaemon(true);
        t.start();
        return t;
    }

    /** OpenAI 流式兼容：JNI 逐 token 回调，经管道流实时下发 SSE；客户端断连即取消生成。 */
    private Response sse(String id, java.util.List<String[]> history) {
        java.io.PipedInputStream in = new java.io.PipedInputStream(1 << 16);
        try {
            final java.io.PipedOutputStream out =
                    new java.io.PipedOutputStream(in);
            Thread worker = new Thread(() -> {
                ThinkFilter filter = new ThinkFilter();
                final boolean[] clientGone = {false};
                Thread wd = watchdog(GEN_TIMEOUT_MS);
                try {
                    llm.generateChatStreaming(history, piece -> {
                        if (clientGone[0]) return;
                        String visible = filter.push(piece);
                        if (visible.isEmpty()) return;
                        try {
                            out.write(sseChunk(id, visible)
                                    .getBytes(StandardCharsets.UTF_8));
                            out.flush();
                        } catch (Exception e) {
                            clientGone[0] = true;
                            AiDebug.i("sse: client disconnected -> cancel generation");
                            llm.cancel();
                        }
                    }, 512, 0.7f);
                    String tail = filter.flush();
                    if (!tail.isEmpty()) {
                        out.write(sseChunk(id, tail).getBytes(StandardCharsets.UTF_8));
                    }
                    out.write(("data: {\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\","
                            + "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
                            .getBytes(StandardCharsets.UTF_8));
                    out.write("data: [DONE]\n\n".getBytes(StandardCharsets.UTF_8));
                    out.flush();
                } catch (Exception e) {
                    AiDebug.e("sse worker: " + e);
                } finally {
                    wd.interrupt();
                    try {
                        out.close();
                    } catch (Exception ignored) {
                    }
                }
            }, "ai-sse");
            worker.start();
            Response r = newChunkedResponse(Response.Status.OK, "text/event-stream", in);
            r.addHeader("Cache-Control", "no-cache");
            r.addHeader("X-Accel-Buffering", "no");
            return r;
        } catch (Exception e) {
            return json(500, err(e.getMessage()));
        }
    }

    private static String sseChunk(String id, String piece) {
        return "data: {\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\","
                + "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" + jsonEscape(piece)
                + "\"}}]}\n\n";
    }

    private static String jsonEscape(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"")
                .replace("\n", "\\n").replace("\r", "");
    }

    private static Response json(int status, String body) {
        Response r = newFixedLengthResponse(Response.Status.lookup(status),
                "application/json", body);
        return r;
    }

    private static String err(String msg) {
        try {
            JSONObject e = new JSONObject();
            e.put("message", msg == null ? "" : msg);
            e.put("type", "local_server_error");
            return new JSONObject().put("error", e).toString();
        } catch (Exception ex) {
            return "{\"error\":{\"message\":\"internal\"}}";
        }
    }
}
