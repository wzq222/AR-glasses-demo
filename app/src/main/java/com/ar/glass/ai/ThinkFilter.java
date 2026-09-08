package com.ar.glass.ai;

/**
 * 思考区流式过滤器：剥离 qwen3 等模型输出的 <think>...</think> 思考内容。
 *
 * 借鉴 F:\Chat2API kernel.ReasoningXMLFilter 的有状态设计：
 * 分片可能在标签中间断开，用"扣留疑似半截前缀"策略跨分片检测；
 * 标签匹配不能用 \b 收尾（半截前缀会被误判为词边界），改用前向界定。
 * flush 时未闭合的 pending 一律丢弃，防止半截标签泄出。
 */
public class ThinkFilter {

    private static final String OPEN = "<think>";
    private static final String CLOSE = "</think>";

    private boolean inThink = false;
    private final StringBuilder pending = new StringBuilder();

    /** 输入一个流式分片，返回可安全展示的文本（可能为空）。 */
    public String push(String piece) {
        StringBuilder out = new StringBuilder();
        String buf = pending + piece;
        pending.setLength(0);

        int i = 0;
        while (i < buf.length()) {
            if (inThink) {
                int close = buf.indexOf(CLOSE, i);
                if (close >= 0) {
                    i = close + CLOSE.length();
                    inThink = false;
                } else {
                    // 尾部可能是半截 </think>，扣住
                    int keep = holdLength(buf, i, CLOSE);
                    if (keep > 0) {
                        pending.append(buf, buf.length() - keep, buf.length());
                    }
                    break;
                }
            } else {
                int open = buf.indexOf(OPEN, i);
                if (open >= 0) {
                    out.append(buf, i, open);
                    i = open + OPEN.length();
                    inThink = true;
                } else {
                    int keep = holdLength(buf, i, OPEN);
                    if (keep > 0) {
                        int emitEnd = buf.length() - keep;
                        out.append(buf, i, emitEnd);
                        pending.append(buf, emitEnd, buf.length());
                    } else {
                        out.append(buf, i, buf.length());
                    }
                    break;
                }
            }
        }
        return out.toString();
    }

    /** 生成结束：扣留内容若已成完整标签则按状态处理，否则丢弃（半截标签不外泄）。 */
    public String flush() {
        String buf = pending.toString();
        pending.setLength(0);
        if (!inThink) {
            if (buf.contains(OPEN)) return "";
            return buf; // 结束时已不可能是半截标签开头，直接吐出
        }
        return ""; // 思考区残留（含半截 </think>）一律丢弃
    }

    /** 返回 buf 尾部可能是 tag 前缀的长度（0 = 无需扣留）。 */
    private static int holdLength(String buf, int from, String tag) {
        for (int len = Math.min(tag.length() - 1, buf.length() - from); len > 0; len--) {
            if (buf.endsWith(tag.substring(0, len))) return len;
        }
        return 0;
    }
}
