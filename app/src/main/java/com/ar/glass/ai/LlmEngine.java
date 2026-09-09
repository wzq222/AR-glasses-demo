package com.ar.glass.ai;

import android.content.Context;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * 本地 LLM 引擎（llama.cpp JNI，Vulkan GPU 加速优先，显存不足自动 CPU 分层回退）。
 *
 * 模型文件从 assets/llm/ 首启动拷贝到 filesDir 后 mmap 加载（assets 内无法 mmap）。
 */
public class LlmEngine {
    private static final String TAG = "AiLlm";

    public static final String MODEL_ASSET = "llm/minicpm5-2b-q4_k_m.gguf";
    public static final String VLM_ASSET = "vlm/minicpm-v4_6-q4_k_m.gguf";
    public static final String MMPROJ_ASSET = "vlm/mmproj-f16.gguf";
    private static final int DEFAULT_CTX = 4096;
    private static final int DEFAULT_GPU_LAYERS = 99; // 全部分层交给 Vulkan，放不下自动回退 CPU

    /**
     * Vulkan 兼容等级（env 必须在 native 库加载前设置，进程内一次生效）：
     * 0=默认；1=禁 coopmat（Mali/Immortalis 驱动 bug，见 llama.cpp#22034/#23057）；
     * 2=再禁 F16（tier1 仍失败时）。
     */
    private static int sVkCompatLevel = -1;
    private static boolean sLibLoaded = false;

    /** 首次使用前调用：检测 GPU 厂商并设置兼容 env，随后加载 native 库。 */
    public static synchronized void initGpuCompat(Context ctx) {
        if (sVkCompatLevel >= 0) return;
        android.content.SharedPreferences sp =
                ctx.getSharedPreferences("ai", Context.MODE_PRIVATE);
        sVkCompatLevel = sp.getInt("vk_compat", -1);
        String renderer = gpuRenderer();
        boolean mali = renderer != null && (renderer.contains("Mali")
                || renderer.contains("Immortalis"));
        if (sVkCompatLevel < 0) sVkCompatLevel = mali ? 1 : 0;
        applyVkEnv(sVkCompatLevel);
        AiDebug.i("vk compat: level=" + sVkCompatLevel + ", gpu=" + renderer
                + (mali ? " (Mali→coopmat 已禁用)" : ""));
        ensureLoaded();
    }

    /** 升级兼容等级并持久化；env 只在下次进程启动生效，返回是否已升级。 */
    public static boolean escalateVkCompat(Context ctx) {
        if (sVkCompatLevel >= 2) return false;
        android.content.SharedPreferences sp =
                ctx.getSharedPreferences("ai", Context.MODE_PRIVATE);
        sp.edit().putInt("vk_compat", sVkCompatLevel + 1).apply();
        AiDebug.i("vk compat escalated to " + (sVkCompatLevel + 1) + "，重启应用后生效");
        return true;
    }

    private static void applyVkEnv(int level) {
        try {
            if (level >= 1) {
                android.system.Os.setenv("GGML_VK_DISABLE_COOPMAT", "1", true);
            }
            if (level >= 2) {
                android.system.Os.setenv("GGML_VK_DISABLE_F16", "1", true);
            }
        } catch (Exception e) {
            AiDebug.e("setenv failed: " + e);
        }
    }

    /** 离屏 EGL 取 GPU renderer 字符串（无需 UI）。 */
    private static String gpuRenderer() {
        android.opengl.EGLDisplay d = null;
        try {
            d = android.opengl.EGL14.eglGetDisplay(android.opengl.EGL14.EGL_DEFAULT_DISPLAY);
            int[] ver = new int[2];
            if (!android.opengl.EGL14.eglInitialize(d, ver, 0, ver, 1)) return "?";
            int[] cfgAttr = {android.opengl.EGL14.EGL_RENDERABLE_TYPE,
                    android.opengl.EGL14.EGL_OPENGL_ES2_BIT, android.opengl.EGL14.EGL_NONE};
            android.opengl.EGLConfig[] cfg = new android.opengl.EGLConfig[1];
            int[] num = new int[1];
            android.opengl.EGL14.eglChooseConfig(d, cfgAttr, 0, cfg, 0, 1, num, 0);
            if (num[0] == 0) return "?";
            android.opengl.EGLContext ec = android.opengl.EGL14.eglCreateContext(d, cfg[0],
                    android.opengl.EGL14.EGL_NO_CONTEXT,
                    new int[]{android.opengl.EGL14.EGL_CONTEXT_CLIENT_VERSION, 2,
                            android.opengl.EGL14.EGL_NONE}, 0);
            android.opengl.EGLSurface es = android.opengl.EGL14.eglCreatePbufferSurface(d,
                    cfg[0], new int[]{android.opengl.EGL14.EGL_WIDTH, 1,
                            android.opengl.EGL14.EGL_HEIGHT, 1,
                            android.opengl.EGL14.EGL_NONE}, 0);
            android.opengl.EGL14.eglMakeCurrent(d, es, es, ec);
            String r = android.opengl.GLES20.glGetString(android.opengl.GLES20.GL_RENDERER);
            android.opengl.EGL14.eglMakeCurrent(d, android.opengl.EGL14.EGL_NO_SURFACE,
                    android.opengl.EGL14.EGL_NO_SURFACE, android.opengl.EGL14.EGL_NO_CONTEXT);
            android.opengl.EGL14.eglDestroySurface(d, es);
            android.opengl.EGL14.eglDestroyContext(d, ec);
            return r == null ? "?" : r;
        } catch (Throwable t) {
            return "?";
        } finally {
            if (d != null) android.opengl.EGL14.eglTerminate(d);
        }
    }

    private static synchronized void ensureLoaded() {
        if (!sLibLoaded) {
            System.loadLibrary("ai_llm");
            sLibLoaded = true;
        }
    }

    private long session = 0;
    private int nCtx = DEFAULT_CTX;

    public interface ProgressListener {
        void onProgress(String file, int percent);
    }

    /** 模型存放根目录：应用外部专属目录（/storage/emulated/0/Android/data/
     *  com.ar.glass/files/），文件管理器可直接查看/删除；不可用时回退内部存储。 */
    public static File modelBaseDir(Context ctx) {
        File ext = ctx.getExternalFilesDir(null);
        return ext != null ? ext : ctx.getFilesDir();
    }

    /** 通用 assets→filesDir 拷贝（幂等），用于 VLM/mmproj 等大文件外置或内置。 */
    public static String ensureAssetFile(Context ctx, String asset) throws Exception {
        File out = new File(modelBaseDir(ctx), asset);
        long assetLen;
        try {
            assetLen = ctx.getAssets().openFd(asset).getLength();
        } catch (Exception e) {
            throw new IllegalStateException("asset not found: " + asset);
        }
        if (out.exists() && out.length() == assetLen) return out.getAbsolutePath();
        File parent = out.getParentFile();
        if (parent != null) parent.mkdirs();
        try (InputStream in = ctx.getAssets().open(asset);
             OutputStream os = new FileOutputStream(out)) {
            byte[] buf = new byte[1 << 20];
            int n;
            while ((n = in.read(buf)) > 0) os.write(buf, 0, n);
        }
        return out.getAbsolutePath();
    }

    /** 确保模型已从 assets 拷出，返回本地文件路径。拷贝一次性，幂等。
     *  GGUF 以 <2GB 分片（.NN.part）打包（AGP assets 单文件上限 2GB），拷贝时顺序拼接。 */
    public static String ensureModelFile(Context ctx, ProgressListener listener)
            throws Exception {
        File out = new File(modelBaseDir(ctx), MODEL_ASSET);
        boolean splitAssets;
        long expected;
        try (InputStream p0 = ctx.getAssets().open(partName(0))) {
            splitAssets = true;
            expected = p0.available() + partsLength(ctx, 1);
        } catch (Exception e) {
            splitAssets = false;
            expected = ctx.getAssets().openFd(MODEL_ASSET).getLength();
        }
        if (out.exists() && out.length() == expected) {
            return out.getAbsolutePath();
        }
        File parent = out.getParentFile();
        if (parent != null) parent.mkdirs();
        try (OutputStream os = new FileOutputStream(out)) {
            if (splitAssets) {
                copyParts(ctx, os, listener, expected);
            } else {
                copySingle(ctx, os, listener, expected);
            }
        }
        if (listener != null) listener.onProgress(MODEL_ASSET, 100);
        return out.getAbsolutePath();
    }

    private static long partsLength(Context ctx, int from) throws Exception {
        long total = 0;
        for (int i = from; ; i++) {
            try (InputStream in = ctx.getAssets().open(partName(i))) {
                total += in.available();
            } catch (Exception e) {
                break;
            }
        }
        return total;
    }

    private static String partName(int i) {
        return String.format("%s.%02d.part", MODEL_ASSET, i);
    }

    private static void copyParts(Context ctx, OutputStream os,
                                  ProgressListener listener, long total) throws Exception {
        long copied = 0;
        byte[] buf = new byte[1 << 20];
        for (int i = 0; ; i++) {
            InputStream in;
            try {
                in = ctx.getAssets().open(partName(i));
            } catch (Exception e) {
                break;
            }
            int n;
            while ((n = in.read(buf)) > 0) {
                os.write(buf, 0, n);
                copied += n;
                if (listener != null && copied % (64 << 20) < n) {
                    listener.onProgress(MODEL_ASSET,
                            total > 0 ? (int) (copied * 100 / total) : 0);
                }
            }
            in.close();
        }
    }

    private static void copySingle(Context ctx, OutputStream os,
                                   ProgressListener listener, long total) throws Exception {
        try (InputStream in = ctx.getAssets().open(MODEL_ASSET)) {
            long copied = 0;
            byte[] buf = new byte[1 << 20];
            int n;
            while ((n = in.read(buf)) > 0) {
                os.write(buf, 0, n);
                copied += n;
                if (listener != null && copied % (64 << 20) < n) {
                    listener.onProgress(MODEL_ASSET,
                            total > 0 ? (int) (copied * 100 / total) : 0);
                }
            }
        }
    }

    /**
     * 加载模型。GPU 优先；加载失败或无 Vulkan 设备时以 CPU 模式重试。
     */
    public synchronized void load(String modelPath) {
        release();
        nCtx = DEFAULT_CTX;
        long bytes = new File(modelPath).length();
        AiDebug.i(String.format("llm load: %s (%.2f GB)", modelPath, bytes / 1e9));
        long t0 = System.currentTimeMillis();
        session = nativeCreateSession(modelPath, nCtx, DEFAULT_GPU_LAYERS, 4);
        if (session == 0) {
            // GPU 路径失败：纯 CPU 重试
            AiDebug.i("gpu-layer load failed, retrying CPU-only (n_gpu_layers=0)");
            session = nativeCreateSession(modelPath, nCtx, 0, 4);
        }
        if (session == 0) {
            AiDebug.e("llm load FAILED: " + modelPath);
            throw new IllegalStateException("LLM model load failed: " + modelPath);
        }
        AiDebug.i(String.format("llm loaded in %d ms, backend=%s",
                System.currentTimeMillis() - t0,
                nativeIsGpuActive() ? "GPU-first(Vulkan)" : "CPU"));
        Log.i(TAG, "model loaded, gpu=" + nativeIsGpuActive());
    }

    public synchronized boolean isLoaded() {
        return session != 0;
    }

    /**
     * 同步生成（内部线程由调用方管理）。返回完整文本；
     * 模板为 MiniCPM5 ChatML；预填空 <think> 块跳过混合思考直接快速响应。
     */
    public synchronized String generate(String userText, int maxTokens, float temperature) {
        if (session == 0) throw new IllegalStateException("model not loaded");
        String prompt = "<|im_start|>user\n" + userText
                + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        return nativeGenerate(session, prompt, maxTokens, temperature);
    }

    /**
     * 多模态生成：图片经 MiniCPM-V 4.6 视觉编码器（mmproj）编码后，
     * 由该 VLM 的语言塔产出回答。prompt 含媒体标记，由本方法套模板。
     */
    public synchronized String generateWithImage(android.graphics.Bitmap image,
                                                 String userText,
                                                 int maxTokens, float temperature) {
        if (session == 0) throw new IllegalStateException("model not loaded");
        String prompt = "<|im_start|>user\n<__image__>" + userText
                + "<|im_end|>\n<|im_start|>assistant\n";
        int w = image.getWidth(), h = image.getHeight();
        int[] px = new int[w * h];
        image.getPixels(px, 0, w, 0, 0, w, h);
        return nativeGenerateWithImage(session, prompt, px, w, h,
                maxTokens, temperature);
    }

    /** 加载 mmproj 视觉投影器（一次性，之后 nativeHasVision 为真）。 */
    public synchronized boolean initVision(String mmprojPath) {
        if (session == 0) throw new IllegalStateException("model not loaded");
        if (nativeHasVision(session)) return true;
        return nativeInitVision(session, mmprojPath);
    }

    /** 流式生成回调。 */
    public interface StreamListener {
        void onToken(String piece);

        void onDone(String fullText);
    }

    public synchronized void generateStreaming(String userText, StreamListener listener,
                                               int maxTokens, float temperature) {
        String full = generate(userText, maxTokens, temperature);
        ThinkFilter filter = new ThinkFilter();
        // 简易分片回放，模拟流式输出并过滤思考区
        int chunk = 8;
        for (int i = 0; i < full.length(); i += chunk) {
            String piece = full.substring(i, Math.min(full.length(), i + chunk));
            String visible = filter.push(piece);
            if (!visible.isEmpty()) listener.onToken(visible);
        }
        String tail = filter.flush();
        if (!tail.isEmpty()) listener.onToken(tail);
        listener.onDone(full);
    }

    public synchronized void release() {
        if (session != 0) {
            nativeFreeSession(session);
            session = 0;
        }
    }

    public native boolean nativeIsGpuActive();

    private native boolean nativeInitVision(long session, String mmprojPath);

    private native boolean nativeHasVision(long session);

    private native String nativeGenerateWithImage(long session, String prompt,
                                                  int[] pixels, int width,
                                                  int height, int maxTokens,
                                                  float temperature);

    private native long nativeCreateSession(String modelPath, int nCtx,
                                            int nGpuLayers, int nThreads);

    private native void nativeFreeSession(long session);

    private native String nativeGenerate(long session, String prompt,
                                         int maxTokens, float temperature);
}
