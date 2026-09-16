package com.ar.glass.ai;

import android.content.Context;
import android.content.SharedPreferences;

import com.ar.glass.ai.model.HotwordStore;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * AI 进程级常驻运行时。
 *
 * <p>把 LLM / ASR / 管线 / 本地 API 服务从「Activity 私有」提升为**进程级单例**，
 * 于是：App 一启动即可自动加载模型并开启 API（无需先进 AI 页面），
 * 且可由 {@link AiDebugReceiver} 用 adb 命令 load / unload / status / gen —— 便于干跑调试与性能评估。
 *
 * <p>所有耗时操作都在内部单线程 worker 上串行执行，避免并发争抢同一个 llama session。
 * 卸载（{@link #unload}）会释放模型与 ASR，用于对比"加载态 vs 卸载态"的内存/功耗基线。
 */
public final class AiRuntime {

    private static final String TAG = "AiRuntime";
    private static final AiRuntime INSTANCE = new AiRuntime();

    public static AiRuntime get() {
        return INSTANCE;
    }

    /** 本地 OpenAI 兼容服务端口。 */
    public static final int API_PORT = 8080;
    public static final String PREFS = "ai";
    /** API 服务是否默认开启（默认 true）。 */
    public static final String PREF_API_ENABLED = "api_enabled";
    /** 启动是否自动加载模型（默认 true）。 */
    public static final String PREF_AUTOLOAD = "autoload";

    private final ExecutorService worker = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "ai-runtime");
        t.setDaemon(true);
        return t;
    });

    private Context app;
    private LlmEngine llm;
    private AsrEngine asr;
    private AiPipeline pipeline;
    private AiHttpServer apiServer;
    private HotwordStore hotwords;

    private volatile boolean loaded = false;
    private volatile String state = "idle";   // idle | loading | ready | error
    private volatile String lastError = "";
    private volatile String llmPath = "";
    private volatile long loadMs = 0;
    private volatile boolean apiEnabled = true;
    private volatile String apiKey = "";

    private AiRuntime() {
    }

    // ================= 对外访问 =================

    public synchronized LlmEngine llm() {
        if (llm == null) llm = new LlmEngine();
        return llm;
    }

    public synchronized AsrEngine asr() {
        if (asr == null) asr = new AsrEngine();
        return asr;
    }

    public synchronized AiPipeline pipeline() {
        if (pipeline == null) pipeline = new AiPipeline(llm());
        return pipeline;
    }

    public HotwordStore hotwords() {
        return hotwords;
    }

    public boolean isLoaded() {
        return loaded;
    }

    public boolean isApiRunning() {
        return apiServer != null && apiServer.wasStarted();
    }

    public String apiKey() {
        return apiKey;
    }

    /** 安装 Application Context 并读取偏好（不发车）。 */
    public synchronized void install(Context ctx) {
        if (app == null) {
            app = ctx.getApplicationContext();
            SharedPreferences sp = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
            apiEnabled = sp.getBoolean(PREF_API_ENABLED, true);
            apiKey = AiHttpServer.getOrCreateApiKey(app);
            AiDebug.i("runtime: installed, apiEnabled=" + apiEnabled);
        }
    }

    /** App 启动时调用：按偏好自动加载模型，并在需要时拉起 API。 */
    public void autoStart(Context ctx) {
        install(ctx);
        SharedPreferences sp = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        boolean autoload = sp.getBoolean(PREF_AUTOLOAD, true);
        AiDebug.i("runtime: autoStart autoload=" + autoload + ", apiEnabled=" + apiEnabled);
        if (autoload && !loaded && !"loading".equals(state)) {
            loadAsync(null);
        } else if (apiEnabled) {
            worker.execute(this::startApiLocked);
        }
    }

    /**
     * 加载模型（异步）。{@code llmPathOrNull} 为空则用内置资产。
     */
    public void loadAsync(String llmPathOrNull) {
        worker.execute(() -> {
            if ("loading".equals(state)) {
                AiDebug.i("runtime: load already in progress, skip");
                return;
            }
            if (loaded && (llmPathOrNull == null || llmPathOrNull.isEmpty())) {
                AiDebug.i("runtime: already loaded, skip duplicate load");
                return;
            }
            try {
                state = "loading";
                lastError = "";
                LlmEngine.initGpuCompat(app);
                LlmEngine engine = llm();
                AsrEngine asrEngine = asr();
                pipeline();   // 确保管线就绪

                String path = llmPathOrNull;
                if (path == null || path.isEmpty()) {
                    path = LlmEngine.ensureModelFile(app, null);
                }
                long t0 = System.currentTimeMillis();
                engine.load(path);
                loadMs = System.currentTimeMillis() - t0;
                llmPath = path;

                asrEngine.load(app);                     // 内部自动探测 hr，缺文件则关闭
                hotwords = HotwordStore.load(app);

                loaded = true;
                state = "ready";
                AiDebug.i(String.format(
                        "runtime: models loaded in %d ms, backend=%s, hotwords=%d",
                        loadMs, engine.nativeIsGpuActive() ? "GPU(Vulkan)" : "CPU",
                        hotwords == null ? 0 : hotwords.allWords().size()));
                if (apiEnabled) startApiLocked();
            } catch (Throwable t) {
                state = "error";
                lastError = String.valueOf(t);
                AiDebug.e("runtime: load failed: " + t);
            }
        });
    }

    /** 卸载模型（释放 LLM/ASR 与 API），用于性能对比基线。 */
    public void unload(String reason) {
        worker.execute(() -> {
            try {
                stopApiLocked();
                if (llm != null) llm.release();
                if (asr != null) asr.release();
                loaded = false;
                state = "idle";
                AiDebug.i("runtime: models unloaded (" + (reason == null ? "-" : reason) + ")");
            } catch (Throwable t) {
                AiDebug.e("runtime: unload failed: " + t);
            }
        });
    }

    /** 开/关 API 服务并持久化偏好。 */
    public void setApiEnabled(boolean enabled) {
        apiEnabled = enabled;
        if (app != null) {
            app.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                    .edit().putBoolean(PREF_API_ENABLED, enabled).apply();
        }
        worker.execute(() -> {
            if (enabled) startApiLocked();
            else stopApiLocked();
        });
    }

    public boolean isApiEnabled() {
        return apiEnabled;
    }

    /** 取消当前生成（调试/客户端断连用）。 */
    public void cancelGen() {
        if (llm != null) {
            try {
                llm.cancel();
            } catch (Throwable t) {
                AiDebug.e("runtime: cancel failed: " + t);
            }
        }
    }

    /**
     * 干跑生成：用于性能参数评估（时延 / 估算 tok/s），结果写 AiLog。
     */
    public void dryRunGen(String text, int maxTokens) {
        final String prompt = (text == null || text.isEmpty()) ? "你好" : text;
        worker.execute(() -> {
            if (!loaded || llm == null) {
                AiDebug.e("runtime: gen skipped (not loaded)");
                return;
            }
            long t0 = System.currentTimeMillis();
            String out = llm.generate(prompt, maxTokens, 0.7f);
            long ms = System.currentTimeMillis() - t0;
            int outTok = AiPipeline.estimateTokens(out);
            float tokPerSec = ms > 0 ? outTok * 1000f / ms : 0f;
            AiDebug.i(String.format(
                    "runtime: gen %d chars -> %d chars (%d tok est) in %d ms, %.1f tok/s(est)",
                    prompt.length(), out.length(), outTok, ms, tokPerSec));
            AiDebug.i("runtime: gen out="
                    + (out.length() > 160 ? out.substring(0, 160) + "..." : out));
        });
    }

    /** 本机局域网 IPv4（取不到返回 null）。 */
    private static String localIpv4() {
        try {
            java.util.Enumeration<java.net.NetworkInterface> ifaces =
                    java.net.NetworkInterface.getNetworkInterfaces();
            while (ifaces != null && ifaces.hasMoreElements()) {
                java.net.NetworkInterface ni = ifaces.nextElement();
                if (!ni.isUp() || ni.isLoopback()) continue;
                java.util.Enumeration<java.net.InetAddress> addrs = ni.getInetAddresses();
                while (addrs.hasMoreElements()) {
                    java.net.InetAddress a = addrs.nextElement();
                    if (a instanceof java.net.Inet4Address && !a.isLoopbackAddress()) {
                        String ip = a.getHostAddress();
                        if (ip != null && !ip.startsWith("127.")) return ip;
                    }
                }
            }
        } catch (Throwable ignored) {
        }
        return null;
    }

    /**
     * 连接信息（可整段复制给调用方）：base_url / api_key / model + curl 示例。
     * 同时给出 127.0.0.1（PC 需先 adb forward）与局域网 IP（同网段可直连）。
     */
    public String connInfo() {
        String key = (apiKey == null || apiKey.isEmpty())
                ? AiHttpServer.DEFAULT_API_KEY : apiKey;
        String local = "http://127.0.0.1:" + API_PORT + "/v1";
        StringBuilder sb = new StringBuilder();
        sb.append("# 本地 OpenAI 兼容服务（AR 眼镜）\n");
        sb.append("base_url: ").append(local).append('\n');
        String ip = localIpv4();
        if (ip != null) {
            sb.append("base_url_lan: http://").append(ip).append(':')
              .append(API_PORT).append("/v1\n");
        }
        sb.append("api_key: ").append(key).append('\n');
        sb.append("model: ").append(AiHttpServer.MODEL_ID).append('\n');
        sb.append("\n# curl 示例（真机需先 adb forward tcp:18080 tcp:")
          .append(API_PORT).append("）\n");
        sb.append("curl ").append(local).append("/chat/completions")
          .append(" -H \"Authorization: Bearer ").append(key).append('"')
          .append(" -H \"Content-Type: application/json\"")
          .append(" -d '{\"model\":\"").append(AiHttpServer.MODEL_ID)
          .append("\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}'");
        return sb.toString();
    }

    /** 运行时状态快照（干跑调试用），单行 key=value。 */
    public String status() {        StringBuilder sb = new StringBuilder();
        String apiState = isApiRunning() ? "running" : (apiEnabled ? "stopped" : "disabled");
        sb.append("state=").append(state)
                .append(" loaded=").append(loaded)
                .append(" api=").append(apiState)
                .append(" port=").append(API_PORT)
                .append(" loadMs=").append(loadMs)
                .append(" gpu=").append(llm != null && llm.nativeIsGpuActive())
                .append(" llmPath=").append(llmPath.isEmpty() ? "(内置)" : llmPath)
                .append(" hotwords=").append(hotwords == null ? 0 : hotwords.allWords().size());
        if (!lastError.isEmpty()) sb.append(" lastError=").append(lastError);
        return sb.toString();
    }

    /** UI 侧手动重载成功后同步运行时状态（供 AiChatActivity 调用）。 */
    public void noteExternalLoad(String path) {
        llmPath = path == null ? "" : path;
        loaded = true;
        state = "ready";
        AiDebug.i("runtime: external load noted, llmPath=" + llmPath);
    }

    /** 进程与内存快照（干跑调试用）。 */
    public String memStatus() {
        try {
            Runtime r = Runtime.getRuntime();
            long heapUsed = (r.totalMemory() - r.freeMemory()) / 1048576;
            long heapMax = r.maxMemory() / 1048576;
            android.os.Debug.MemoryInfo mi = new android.os.Debug.MemoryInfo();
            android.os.Debug.getMemoryInfo(mi);
            return String.format("pid=%d javaHeap=%d/%dMB pss=%dMB nativeHeap=%dMB",
                    android.os.Process.myPid(), heapUsed, heapMax,
                    mi.getTotalPss() / 1024, android.os.Debug.getNativeHeapAllocatedSize() / 1048576);
        } catch (Throwable t) {
            return "mem=n/a (" + t + ")";
        }
    }

    // ================= 内部 =================

    private void startApiLocked() {
        try {
            if (apiServer != null && apiServer.wasStarted()) {
                apiServer.markReady();
                return;
            }
            if (llm == null || pipeline == null) {
                AiDebug.i("runtime: api deferred (engines not ready)");
                return;
            }
            if (apiKey == null || apiKey.isEmpty()) apiKey = AiHttpServer.getOrCreateApiKey(app);
            apiServer = new AiHttpServer(app, API_PORT, llm, pipeline);
            apiServer.setKeyProvider(() -> apiKey);
            apiServer.start(5000, true);
            apiServer.markReady();
            AiDebug.i("[API] started http://127.0.0.1:" + API_PORT + "/v1  apiKey=" + apiKey);
        } catch (Throwable t) {
            AiDebug.e("[API] start failed: " + t);
        }
    }

    private void stopApiLocked() {
        try {
            if (apiServer != null) {
                apiServer.stop();
                apiServer = null;
                AiDebug.i("[API] stopped");
            }
        } catch (Throwable t) {
            AiDebug.e("[API] stop failed: " + t);
        }
    }
}
