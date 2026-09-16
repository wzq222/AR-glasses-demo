package com.ar.glass.ai.model;

import android.content.Context;
import android.util.Log;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 热词 / 唤醒词字典（预留）。
 *
 * <p>来源优先级：设备可写目录 {@code <modelBaseDir>/ai/hotwords.txt}（用户可改）
 * &gt; 内置 assets {@code ai/hotwords.txt}（随包默认）。
 * 改设备侧文件即可生效，<b>无需重新编译或重装</b>。
 *
 * <p>文件格式（见 assets/ai/hotwords.txt 头部注释）：
 * <pre>
 *   [wake]        唤醒词段（后续语音唤醒用）
 *   你好眼镜
 *   [term]        领域术语段（用于生成 hr 同音字替换规则 / 导出 sherpa 热词表）
 *   紧固件 :2.0   （":权重" 可省略）
 * </pre>
 *
 * <p>说明：本类只负责"读取与暴露"字典，不绑定具体消费方——
 * 唤醒词可供后续 KWS(KeywordSpotter) 或转写文本匹配使用；
 * 术语段可导出为 sherpa 兼容热词表，或用于生成 hr 的 lexicon/replace.fst。
 */
public final class HotwordStore {

    private static final String TAG = "AiHotword";

    /** 内置字典的 assets 路径。 */
    public static final String ASSET_PATH = "ai/hotwords.txt";
    /** 设备侧覆盖文件（相对 modelBaseDir）。 */
    public static final String USER_REL_PATH = "ai/hotwords.txt";
    /** 导出的 sherpa 兼容热词表（相对 modelBaseDir）。 */
    public static final String SHERPA_REL_PATH = "ai/hotwords.sherpa.txt";

    private final List<String> wakeRaw = new ArrayList<>();
    private final List<String> termRaw = new ArrayList<>();
    private String source = "";

    private HotwordStore() {
    }

    /** 读取字典：优先设备侧文件，其次内置 assets。任何异常都退化为空字典，不影响 ASR。 */
    public static HotwordStore load(Context ctx) {
        HotwordStore store = new HotwordStore();
        String text = null;
        try {
            File user = new File(baseDir(ctx), USER_REL_PATH);
            if (user.isFile() && user.length() > 0) {
                store.source = user.getAbsolutePath();
                text = readFile(user);
            }
            if (text == null) {
                store.source = "assets:" + ASSET_PATH;
                text = readAsset(ctx, ASSET_PATH);
            }
        } catch (Exception e) {
            Log.w(TAG, "load hotwords failed: " + e);
        }
        store.parse(text);
        Log.i(TAG, "hotwords loaded from " + store.source
                + ": wake=" + store.wakeRaw.size() + ", term=" + store.termRaw.size());
        return store;
    }

    private void parse(String text) {
        if (text == null) return;
        int section = 2; // 默认归入 term
        for (String raw : text.split("\r?\n")) {
            String line = raw.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            if (line.startsWith("[")) {
                section = line.toLowerCase().startsWith("[wake") ? 1 : 2;
                continue;
            }
            if (section == 1) wakeRaw.add(line);
            else termRaw.add(line);
        }
    }

    /** 唤醒词（已剥离 ":权重" 后缀）。 */
    public List<String> wakeWords() {
        return heads(wakeRaw);
    }

    /** 领域术语（已剥离 ":权重" 后缀）。 */
    public List<String> terms() {
        return heads(termRaw);
    }

    /** 全部词条（唤醒词 + 术语）。 */
    public List<String> allWords() {
        List<String> all = new ArrayList<>(wakeWords());
        all.addAll(terms());
        return Collections.unmodifiableList(all);
    }

    public boolean isEmpty() {
        return wakeRaw.isEmpty() && termRaw.isEmpty();
    }

    /** 字典实际来源：设备文件绝对路径，或 "assets:ai/hotwords.txt"。 */
    public String source() {
        return source;
    }

    /**
     * 唤醒命中检测：text 中包含任一唤醒词即返回该唤醒词，否则返回 null。
     * 供后续语音唤醒（文本侧匹配）使用；KWS 硬件侧方案见 docs/AI-MODULARIZATION.md。
     */
    public String matchWake(String text) {
        if (text == null || text.isEmpty()) return null;
        for (String w : wakeWords()) {
            if (!w.isEmpty() && text.contains(w)) return w;
        }
        return null;
    }

    /**
     * 导出 sherpa 兼容热词表（每行一个词条，保留 ":权重" 写法）。
     * 仅当字典非空时写文件；返回绝对路径，字典为空返回 null。
     */
    public String exportSherpaFile(Context ctx) {
        if (isEmpty()) return null;
        try {
            File out = new File(baseDir(ctx), SHERPA_REL_PATH);
            File parent = out.getParentFile();
            if (parent != null) parent.mkdirs();
            try (Writer w = new OutputStreamWriter(
                    new FileOutputStream(out), StandardCharsets.UTF_8)) {
                for (String line : wakeRaw) w.write(line + "\n");
                for (String line : termRaw) w.write(line + "\n");
            }
            return out.getAbsolutePath();
        } catch (Exception e) {
            Log.w(TAG, "export sherpa hotwords failed: " + e);
            return null;
        }
    }

    private static List<String> heads(List<String> raw) {
        List<String> out = new ArrayList<>(raw.size());
        for (String line : raw) {
            String head = line;
            int colon = line.indexOf(':');
            if (colon > 0) head = line.substring(0, colon).trim();
            if (!head.isEmpty()) out.add(head);
        }
        return Collections.unmodifiableList(out);
    }

    private static File baseDir(Context ctx) {
        File ext = ctx.getExternalFilesDir(null);
        return ext != null ? ext : ctx.getFilesDir();
    }

    private static String readAsset(Context ctx, String path) throws Exception {
        try (InputStream in = ctx.getAssets().open(path)) {
            return readAll(in);
        }
    }

    private static String readFile(File f) throws Exception {
        try (InputStream in = new FileInputStream(f)) {
            return readAll(in);
        }
    }

    private static String readAll(InputStream in) throws Exception {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader r = new BufferedReader(
                new InputStreamReader(in, StandardCharsets.UTF_8))) {
            String line;
            while ((line = r.readLine()) != null) {
                sb.append(line).append('\n');
            }
        }
        return sb.toString();
    }
}
