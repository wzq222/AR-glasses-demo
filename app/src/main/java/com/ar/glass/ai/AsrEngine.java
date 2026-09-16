package com.ar.glass.ai;

import android.content.Context;
import android.util.Log;

import com.k2fsa.sherpa.onnx.HomophoneReplacerConfig;
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

    /** hr（同音字替换）可选资产：lexicon 与 replace.fst 必须同时存在才启用，缺一则保持关闭。 */
    public static final String HR_LEXICON_ASSET = "ai/hr/lexicon.txt";
    public static final String HR_FST_ASSET = "ai/hr/replace.fst";
    private static final String HR_DIR = "ai/hr";

    /**
     * ASR 高级选项（全部可选；留空即关闭对应能力，保证默认行为与改造前完全一致）。
     *
     * <ul>
     *   <li>{@code hotwordsFile} —— sherpa 热词表。⚠️ 热词仅对 Transducer/Paraformer 解码器生效，
     *       SenseVoice(CTC) 不消费它，因此默认不设置；留给将来换模型时启用
     *       （词表可用 {@code HotwordStore.exportSherpaFile()} 生成）。</li>
     *   <li>{@code hrLexicon}/{@code hrRuleFsts}/{@code hrDictDir} —— 同音字替换（SenseVoice 可用，
     *       官方支持见 sherpa-onnx PR #2153）。确定性、零算力，可进实时路径。</li>
     * </ul>
     */
    public static class Options {
        public String hotwordsFile;
        public float hotwordsScore = 1.5f;
        public String hrLexicon;
        public String hrRuleFsts;
        public String hrDictDir;
    }

    private OfflineRecognizer recognizer;

    /** SenseVoice 两个小文件直接从 assets 读也可以（免压缩），但 onnx runtime
     *  需要文件路径，统一直接拷到 filesDir。 */
    public static void ensureModelFiles(Context ctx) throws Exception {
        copyAssetIfNeeded(ctx, MODEL_ASSET);
        copyAssetIfNeeded(ctx, TOKENS_ASSET);
    }

    private static void copyAssetIfNeeded(Context ctx, String asset) throws Exception {
        File out = new File(LlmEngine.modelBaseDir(ctx), asset);
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
        File base = new File(LlmEngine.modelBaseDir(ctx), "asr");
        load(new File(base, "sensevoice-small-int8.onnx").getPath(),
                new File(base, "tokens.txt").getPath(), "zh", resolveHr(ctx));
    }

    /**
     * 外部模型接入接口：自定义 SenseVoice onnx / tokens 路径与语言。
     * 模型须为 sherpa-onnx 兼容的 SenseVoice ONNX（含元数据）。
     */
    public synchronized void load(String modelPath, String tokensPath, String language) {
        load(modelPath, tokensPath, language, new Options());
    }

    /** 带高级选项的加载：可选热词 / 可选同音字替换（hr）。 */
    public synchronized void load(String modelPath, String tokensPath, String language,
                                  Options opt) {
        release();
        OfflineRecognizerConfig.Builder builder = OfflineRecognizerConfig.builder()
                .setOfflineModelConfig(OfflineModelConfig.builder()
                        .setSenseVoice(OfflineSenseVoiceModelConfig.builder()
                                .setModel(modelPath)
                                .setLanguage(language == null || language.isEmpty() ? "zh" : language)
                                .setInverseTextNormalization(true)
                                .build())
                        .setTokens(tokensPath)
                        .setNumThreads(2)
                        .setDebug(false)
                        .setProvider("cpu")
                        .setModelType("sense_voice_ctc")
                        .build())
                .setDecodingMethod("greedy_search");
        if (opt != null) {
            if (notEmpty(opt.hrLexicon) && notEmpty(opt.hrRuleFsts)) {
                builder.setHr(HomophoneReplacerConfig.builder()
                        .setLexicon(opt.hrLexicon)
                        .setRuleFsts(opt.hrRuleFsts)
                        .setDictDir(opt.hrDictDir == null ? "" : opt.hrDictDir)
                        .build());
                AiDebug.i("asr hr enabled: lexicon=" + opt.hrLexicon
                        + ", ruleFsts=" + opt.hrRuleFsts);
            }
            if (notEmpty(opt.hotwordsFile)) {
                builder.setHotwordsFile(opt.hotwordsFile).setHotwordsScore(opt.hotwordsScore);
                AiDebug.i("asr hotwords enabled: " + opt.hotwordsFile
                        + " (score=" + opt.hotwordsScore + ")");
            }
        }
        recognizer = new OfflineRecognizer(builder.build());
        AiDebug.i("asr loaded: model=" + modelPath + ", tokens=" + tokensPath
                + ", lang=" + language);
        Log.i(TAG, "SenseVoice recognizer loaded: " + modelPath);
    }

    /**
     * 解析 hr（同音字替换）文件：优先设备可写目录 {@code files/ai/hr/}；缺失时，
     * 仅当内置 assets 同时具备 lexicon 与 replace.fst 才拷出启用。
     * 任一缺失 → 返回空 Options（hr 关闭，ASR 行为不变）。
     */
    private static Options resolveHr(Context ctx) {
        Options opt = new Options();
        try {
            File dir = new File(LlmEngine.modelBaseDir(ctx), HR_DIR);
            File lex = new File(dir, "lexicon.txt");
            File fst = new File(dir, "replace.fst");
            if (!(lex.isFile() && fst.isFile())
                    && assetExists(ctx, HR_LEXICON_ASSET) && assetExists(ctx, HR_FST_ASSET)) {
                File aLex = ensureAssetQuiet(ctx, HR_LEXICON_ASSET);
                File aFst = ensureAssetQuiet(ctx, HR_FST_ASSET);
                if (aLex != null && aFst != null) {
                    lex = aLex;
                    fst = aFst;
                }
            }
            if (lex.isFile() && fst.isFile()) {
                opt.hrLexicon = lex.getAbsolutePath();
                opt.hrRuleFsts = fst.getAbsolutePath();
                opt.hrDictDir = lex.getParentFile() == null
                        ? "" : lex.getParentFile().getAbsolutePath();
            } else {
                AiDebug.i("asr hr disabled (lexicon/replace.fst 未就位)");
            }
        } catch (Exception e) {
            AiDebug.i("asr hr resolve failed: " + e);
        }
        return opt;
    }

    private static boolean notEmpty(String s) {
        return s != null && !s.isEmpty();
    }

    private static boolean assetExists(Context ctx, String asset) {
        try (InputStream in = ctx.getAssets().open(asset)) {
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    /** 从 assets 幂等拷出；资产不存在或拷贝失败返回 null（不抛异常）。 */
    private static File ensureAssetQuiet(Context ctx, String asset) {
        try {
            File out = new File(LlmEngine.modelBaseDir(ctx), asset);
            if (out.isFile() && out.length() > 0) return out;
            File parent = out.getParentFile();
            if (parent != null) parent.mkdirs();
            try (InputStream in = ctx.getAssets().open(asset);
                 OutputStream os = new FileOutputStream(out)) {
                byte[] buf = new byte[1 << 16];
                int n;
                while ((n = in.read(buf)) > 0) os.write(buf, 0, n);
            }
            return out.isFile() && out.length() > 0 ? out : null;
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * 转写 16kHz 单声道 PCM（[-1,1) 浮点）。
     */
    public synchronized String transcribe(float[] samples) {
        if (recognizer == null) throw new IllegalStateException("ASR not loaded");
        long t0 = android.os.SystemClock.elapsedRealtime();
        OfflineStream stream = recognizer.createStream();
        stream.acceptWaveform(samples, SAMPLE_RATE);
        recognizer.decode(stream);
        String text = recognizer.getResult(stream).getText();
        stream.release();
        text = text == null ? "" : text.trim();
        float secs = samples.length / (float) SAMPLE_RATE;
        AiDebug.i(String.format(
                "asr: %.1fs audio -> %d chars in %d ms (%.2fx RTF): %s",
                secs, text.length(),
                android.os.SystemClock.elapsedRealtime() - t0,
                secs > 0 ? (android.os.SystemClock.elapsedRealtime() - t0) / 1000f / secs : 0f,
                text.isEmpty() ? "(空)" : text));
        return text;
    }

    public synchronized void release() {
        if (recognizer != null) {
            recognizer.release();
            recognizer = null;
        }
    }
}
