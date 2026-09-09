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

    public static final String MODEL_ASSET = "llm/qwen3-4b-q4_k_m.gguf";
    private static final int DEFAULT_CTX = 4096;
    private static final int DEFAULT_GPU_LAYERS = 99; // 全部分层交给 Vulkan，放不下自动回退 CPU

    static {
        System.loadLibrary("ai_llm");
    }

    private long session = 0;
    private int nCtx = DEFAULT_CTX;

    public interface ProgressListener {
        void onProgress(String file, int percent);
    }

    /** 确保模型已从 assets 拷出，返回本地文件路径。拷贝一次性，幂等。
     *  GGUF 以 <2GB 分片（.NN.part）打包（AGP assets 单文件上限 2GB），拷贝时顺序拼接。 */
    public static String ensureModelFile(Context ctx, ProgressListener listener)
            throws Exception {
        File out = new File(ctx.getFilesDir(), MODEL_ASSET);
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
     * 模板为 Qwen3 ChatML，enable_thinking 关闭以抑制思考输出。
     */
    public synchronized String generate(String userText, int maxTokens, float temperature) {
        if (session == 0) throw new IllegalStateException("model not loaded");
        String prompt = "<|im_start|>user\n" + userText
                + "<|im_end|>\n<|im_start|>assistant\n";
        return nativeGenerate(session, prompt, maxTokens, temperature);
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

    private native long nativeCreateSession(String modelPath, int nCtx,
                                            int nGpuLayers, int nThreads);

    private native void nativeFreeSession(long session);

    private native String nativeGenerate(long session, String prompt,
                                         int maxTokens, float temperature);
}
