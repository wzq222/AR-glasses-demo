package com.ar.glass.ai;

import android.util.Log;

/**
 * AI 管线：ASR 转写 → LLM 修正 → 结构化摘要。
 *
 * Prompt 移植自 PC 端 bilibili.bat 转写管线；上下文管理借鉴 F:\Chat2API 的
 * 消息归一化思路（对本地无状态调用简化为单轮 + 长文本分块，块间无交叉依赖，
 * 失败重试不影响其它块）。
 */
public class AiPipeline {

    /** 长文本分块上限：约 2500 字符，避免超过本地 ctx。 */
    private static final int MAX_CHARS = 2500;
    /**
     * 无 tokenizer 时的轻量估算：中文约 1.5 字符/token。
     * （旧值 2.5 是英文经验值，会把中文 token 数低估约 2×，导致 usage / tok-s 系统性偏低。）
     */
    private static final double CHARS_PER_TOKEN = 1.5;

    private final LlmEngine llm;

    public AiPipeline(LlmEngine llm) {
        this.llm = llm;
    }

    public static int estimateTokens(String text) {
        return text == null ? 0 : (int) Math.ceil(text.length() / CHARS_PER_TOKEN);
    }

    /** LLM 修正 ASR 原文：断句、标点、同音字纠错、删广告。长文本自动分块。 */
    public String correct(String raw) {
        if (raw == null || raw.trim().isEmpty()) return "";
        if (raw.length() <= MAX_CHARS) return llm.generate(promptCorrect(raw), 1024, 0.3f);
        StringBuilder out = new StringBuilder();
        java.util.List<String> chunks = chunk(raw);
        Log.i("AiPipeline", "correct chunks=" + chunks.size());
        for (int i = 0; i < chunks.size(); i++) {
            out.append(llm.generate(promptCorrect(chunks.get(i)), 1024, 0.3f));
            out.append("\n\n");
        }
        return out.toString().trim();
    }

    /** 结构化摘要：核心观点 / 关键要点 / 结论。 */
    public String summarize(String text) {
        if (text == null || text.trim().isEmpty()) return "";
        return llm.generate(promptSummarize(text), 512, 0.3f);
    }

    /** 直接对话。 */
    public String chat(String userText) {
        return llm.generate(userText, 1024, 0.7f);
    }

    public static String promptCorrect(String text) {
        return "你是中文语音识别结果修正专家。下面是 ASR 转写的原始文本，存在断句错误、"
                + "标点缺失、同音字误识别、上下文不通顺等问题。\n\n"
                + "修正要求：\n"
                + "1. 重新断句，使每句完整通顺，符合中文表达习惯\n"
                + "2. 根据上下文推测并修正同音字、近音字误识别\n"
                + "3. 补全缺失的标点符号\n"
                + "4. 删除明显的广告插入内容，用 [广告已删] 标记位置\n"
                + "5. 保留原文意思，不要添加原文没有的信息\n"
                + "6. 直接输出修正后的纯文本，不要任何解释或前后缀\n\n"
                + "原文：\n" + text;
    }

    public static String promptSummarize(String text) {
        return "请为以下内容生成结构化摘要：\n\n"
                + "要求：\n"
                + "1. 【核心观点】用 2-3 句话概括主旨\n"
                + "2. 【关键要点】用 - 列出 3-6 个关键论点或事实\n"
                + "3. 【结论】用 1 句话总结立场或趋势判断\n"
                + "4. 客观中立，不添加原文没有的信息\n"
                + "5. 直接输出摘要，不要任何解释或前后缀\n\n"
                + "内容：\n" + text;
    }

    /** 按段落边界分块，块长不超过 MAX_CHARS。 */
    static java.util.List<String> chunk(String text) {
        java.util.List<String> chunks = new java.util.ArrayList<>();
        String cur = "";
        for (String p : text.split("\n+")) {
            if (cur.length() + p.length() + 1 > MAX_CHARS) {
                if (!cur.isEmpty()) chunks.add(cur);
                cur = p;
            } else {
                cur = cur.isEmpty() ? p : cur + "\n" + p;
            }
        }
        if (!cur.isEmpty()) chunks.add(cur);
        return chunks;
    }
}
