package com.ar.glass.ai;

import android.content.Context;
import android.util.Log;

import com.k2fsa.sherpa.onnx.OfflineModelConfig;
import com.k2fsa.sherpa.onnx.OfflineRecognizer;
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig;
import com.k2fsa.sherpa.onnx.OfflineSenseVoiceModelConfig;
import com.k2fsa.sherpa.onnx.OfflineStream;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * 本地 ASR 引擎：sherpa-onnx + SenseVoiceSmall（int8 ONNX）。
 * 输入 16kHz 单声道 PCM，输出中文（可切 en/yue/ja/ko/auto）。
 */
public class AsrEngine {
    private static final String TAG = "AiAsr";

    public static final String MODEL_ASSET = "asr/sensevoice-small-int8.onnx";
    public static final String TOKENS_ASSET = "asr/tokens.txt";
    public static final int SAMPLE_RATE = 16000;

    private OfflineRecognizer recognizer;

    /** SenseVoice 两个小文件直接从 assets 读也可以（免压缩），但 onnx runtime
     *  需要文件路径，统一直接拷到 filesDir。 */
    public static void ensureModelFiles(Context ctx) throws Exception {
        copyAssetIfNeeded(ctx, MODEL_ASSET);
        copyAssetIfNeeded(ctx, TOKENS_ASSET);
    }

    private static void copyAssetIfNeeded(Context ctx, String asset) throws Exception {
        File out = new File(ctx.getFilesDir(), asset);
        if (out.exists() && out.length() > 0) return;
        File parent = out.getParentFile();
        if (parent != null) parent.mkdirs();
        try (InputStream in = ctx.getAssets().open(asset);
             OutputStream os = new FileOutputStream(out)) {
            byte[] buf = new byte[1 << 16];
            int n;
            while ((n = in.read(buf)) > 0) os.write(buf, 0, n);
        }
    }

    public synchronized void load(Context ctx) throws Exception {
        release();
        ensureModelFiles(ctx);
        File base = new File(ctx.getFilesDir(), "asr");
        OfflineRecognizerConfig config = OfflineRecognizerConfig.builder()
                .setOfflineModelConfig(OfflineModelConfig.builder()
                        .setSenseVoice(OfflineSenseVoiceModelConfig.builder()
                                .setModel(new File(base, "sensevoice-small-int8.onnx").getPath())
                                .setLanguage("zh")
                                .setInverseTextNormalization(true)
                                .build())
                        .setTokens(new File(base, "tokens.txt").getPath())
                        .setNumThreads(2)
                        .setDebug(false)
                        .setProvider("cpu")
                        .setModelType("sense_voice_ctc")
                        .build())
                .setDecodingMethod("greedy_search")
                .build();
        recognizer = new OfflineRecognizer(config);
        Log.i(TAG, "SenseVoice recognizer loaded");
    }

    /**
     * 转写 16kHz 单声道 PCM（[-1,1) 浮点）。
     */
    public synchronized String transcribe(float[] samples) {
        if (recognizer == null) throw new IllegalStateException("ASR not loaded");
        OfflineStream stream = recognizer.createStream();
        stream.acceptWaveform(samples, SAMPLE_RATE);
        recognizer.decode(stream);
        String text = recognizer.getResult(stream).getText();
        stream.release();
        return text == null ? "" : text.trim();
    }

    public synchronized void release() {
        if (recognizer != null) {
            recognizer.release();
            recognizer = null;
        }
    }
}
