package com.ar.glass.ai;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.ar.glass.R;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * AI 助手简易界面：文本对话 + 按住说话（ASR）+ 修正/摘要快捷操作。
 * 仅作后端管线的前端验证，不做复杂 UI。
 */
public class AiChatActivity extends AppCompatActivity {

    private static final int REQ_RECORD_AUDIO = 101;
    private static final int REQ_PICK_IMAGE = 102;
    /** 流式分段的重叠窗口（0.3s）：给下一段带一点上文音频，减少段边界截词。 */
    private static final int OVERLAP_SAMPLES = AsrEngine.SAMPLE_RATE * 3 / 10;

    private TextView logView;
    private EditText input, llmPathInput, asrPathInput;
    private Button btnSend, btnTalk, btnCorrect, btnSummarize, btnClear, btnReload, btnImage;
    private Button btnApi;
    private Button btnCopyApi;
    private android.graphics.Bitmap pendingImage;

    private LlmEngine llm;
    private AsrEngine asr;
    private AiPipeline pipeline;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    private volatile AudioRecord recorder;
    private volatile boolean recording = false;

    private final StringBuilder transcript = new StringBuilder();

    @SuppressLint("ClickableViewAccessibility")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_ai_chat);

        logView = findViewById(R.id.ai_log);
        logView.setMovementMethod(new ScrollingMovementMethod());
        input = findViewById(R.id.ai_input);
        llmPathInput = findViewById(R.id.ai_llm_path);
        asrPathInput = findViewById(R.id.ai_asr_path);
        btnSend = findViewById(R.id.ai_send);
        btnTalk = findViewById(R.id.ai_talk);
        btnCorrect = findViewById(R.id.ai_correct);
        btnSummarize = findViewById(R.id.ai_summarize);
        btnClear = findViewById(R.id.ai_clear);
        btnReload = findViewById(R.id.ai_reload);
        btnImage = findViewById(R.id.ai_image);
        btnApi = findViewById(R.id.ai_api);
        btnCopyApi = findViewById(R.id.ai_copyapi);

        // 引擎来自进程级常驻运行时（App 启动即自动加载 + 常开本地 API），
        // 这里只取引用，避免同一模型被加载两份。
        AiRuntime rt = AiRuntime.get();
        rt.install(this);
        llm = rt.llm();
        asr = rt.asr();
        pipeline = rt.pipeline();

        log("引擎初始化中（首次加载 LLM 约需 30-60s）…");
        rt.autoStart(this);   // 幂等：已在加载中则忽略
        executor.execute(() -> {
            try { Thread.sleep(1200); } catch (InterruptedException ignored) { }
            runOnUiThread(() -> log("[状态] " + AiRuntime.get().status()));
        });

        // 本地 API 服务开关：OpenAI 兼容，端口+API_KEY，便于云端/本地切换
        btnApi.setOnClickListener(v -> toggleApiServer());
        // 一键把 base_url / api_key / model / curl 示例复制到剪贴板
        btnCopyApi.setOnClickListener(v -> copyConnInfo());
        btnApi.setOnLongClickListener(v -> {
            copyConnInfo();
            return true;
        });

        // 模型接口：LLM 填 GGUF 文件路径、ASR 填 SenseVoice onnx 路径（tokens.txt 同目录），
        // 留空则使用内置模型；修改后点击"重载模型"生效
        btnReload.setOnClickListener(v -> {
            log("重载模型中…");
            // 在主线程读取输入框（后台线程读 EditText 属 UI 线程违规）
            final String lp = llmPathInput.getText().toString().trim();
            final String ap = asrPathInput.getText().toString().trim();
            executor.execute(() -> reloadModels(lp, ap));
        });

        btnSend.setOnClickListener(v -> {
            String text = input.getText().toString().trim();
            android.graphics.Bitmap img = pendingImage != null
                    ? pendingImage : AiDebugBus.pendingImage;
            if (text.isEmpty() && img == null) return;
            input.setText("");
            pendingImage = null;
            AiDebugBus.pendingImage = null;
            final String prompt = text.isEmpty() ? "描述这张图片" : text;
            final android.graphics.Bitmap sendImg = img;
            runAsync(() -> {
                if (sendImg != null) {
                    append("我: [图片] " + prompt);
                    // 懒加载视觉投影器（一次性），随后走 VLM 图文通路
                    String mmproj = vlmPathFromAssets();
                    if (mmproj == null || !llm.initVision(mmproj)) {
                        appendVisible("AI: [视觉编码器加载失败]");
                        return;
                    }
                    String reply = llm.generateWithImage(sendImg, prompt, 512, 0.7f);
                    appendVisible("AI: " + reply);
                } else {
                    append("我: " + text);
                    String reply = pipeline.chat(text);
                    appendVisible("AI: " + reply);
                }
            });
        });

        // 图片选择：系统相册，选图后随下一条消息发送
        btnImage.setOnClickListener(v -> {
            android.content.Intent i = new android.content.Intent(
                    android.content.Intent.ACTION_GET_CONTENT);
            i.setType("image/*");
            startActivityForResult(android.content.Intent.createChooser(i, "选择图片"),
                    REQ_PICK_IMAGE);
        });

        // 按住说话：按下开始录音，松开转写并显示
        btnTalk.setOnTouchListener((v, ev) -> {
            switch (ev.getActionMasked()) {
                case MotionEvent.ACTION_DOWN:
                    startRecording();
                    return true;
                case MotionEvent.ACTION_UP:
                case MotionEvent.ACTION_CANCEL:
                    stopRecordingAndTranscribe();
                    return true;
            }
            return false;
        });

        btnCorrect.setOnClickListener(v -> {
            String src = transcript.toString().trim();
            if (src.isEmpty()) {
                toast("没有可修正的内容");
                return;
            }
            runAsync(() -> {
                append("（修正中…）");
                String fixed = pipeline.correct(src);
                transcript.setLength(0);
                transcript.append(fixed);
                appendVisible("修正:\n" + fixed);
            });
        });

        btnSummarize.setOnClickListener(v -> {
            String src = transcript.toString().trim();
            if (src.isEmpty()) {
                toast("没有可摘要的内容");
                return;
            }
            runAsync(() -> {
                append("（摘要中…）");
                appendVisible("摘要:\n" + pipeline.summarize(src));
            });
        });

        btnClear.setOnClickListener(v -> {
            transcript.setLength(0);
            logView.setText("");
        });
    }

    /**
     * 按给定路径重载 LLM/ASR（在后台线程调用）。路径留空用内置模型。
     * 入参由主线程预先从输入框取出，避免后台线程读 EditText。
     */
    private void reloadModels(String llmPathIn, String asrPathIn) {
        AiDebug.deviceSnapshot();
        AiDebug.i("reload: llmPath=" + llmPathIn + " (空=内置), asrPath=" + asrPathIn + " (空=内置)");
        try {
            String llmPath = llmPathIn;
            if (llmPath.isEmpty()) {
                llmPath = LlmEngine.ensureModelFile(this, null);
            }
            final String llmPathFinal = llmPath;
            if (!new java.io.File(llmPathFinal).exists()) {
                runOnUiThread(() -> log("[ERR] LLM 路径不存在: " + llmPathFinal));
                return;
            }
            llm.load(llmPathFinal);

            String asrPath = asrPathIn;
            if (asrPath.isEmpty()) {
                asr.load(this);
            } else {
                if (!android.os.Environment.isExternalStorageManager()) {
                    runOnUiThread(this::requestAllFilesAccess);
                    return;
                }
                java.io.File onnx = new java.io.File(asrPath);
                java.io.File tokens = new java.io.File(onnx.getParentFile(), "tokens.txt");
                if (!onnx.exists() || !tokens.exists()) {
                    runOnUiThread(() -> log("[ERR] ASR 路径无效（需 onnx + 同目录 tokens.txt）"));
                    return;
                }
                asr.load(onnx.getPath(), tokens.getPath(), "zh");
            }
            AiRuntime.get().noteExternalLoad(llmPathFinal);
            boolean gpu = llm.nativeIsGpuActive();
            AiDebug.i("engines ready: backend=" + (gpu ? "GPU(Vulkan)" : "CPU")
                    + ", llm=" + llmPathFinal);
            if (gpu) {
                runOnUiThread(() -> log("[OK] 引擎就绪 后端=GPU(Vulkan)"));
            } else {
                // 有 GPU 设备但被后端拒绝/回退：升级兼容等级，下次启动重试
                boolean escalated = LlmEngine.escalateVkCompat(this);
                runOnUiThread(() -> log("[WARN] Vulkan 初始化失败，已回退 CPU"
                        + (escalated ? "；已自动升级兼容等级，请退出应用重新打开重试"
                        : "（兼容等级已到顶）")));
            }
        } catch (Throwable e) {
            AiDebug.e("model load failed: " + e);
            runOnUiThread(() -> log("[ERR] 模型加载失败: " + e.getMessage()));
        }
    }

    /** 拷出 VLM/mmproj 资产（懒加载，仅在首次发图时执行）。 */
    private String vlmPathFromAssets() {
        try {
            LlmEngine.ensureAssetFile(this, LlmEngine.VLM_ASSET);
            return LlmEngine.ensureAssetFile(this, LlmEngine.MMPROJ_ASSET);
        } catch (Exception e) {
            AiDebug.e("vlm asset copy failed: " + e);
            return null;
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode,
                                    android.content.Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQ_PICK_IMAGE && resultCode == RESULT_OK && data != null
                && data.getData() != null) {
            executor.execute(() -> {
                try {
                    android.graphics.Bitmap bm = android.graphics.ImageDecoder
                            .decodeBitmap(android.graphics.ImageDecoder
                                    .createSource(getContentResolver(), data.getData()),
                                    (decoder, info, src) -> {
                                        int maxSide = 896; // 控制视觉 token 量
                                        int w = info.getSize().getWidth();
                                        int h = info.getSize().getHeight();
                                        float scale = Math.min(1f,
                                                maxSide / (float) Math.max(w, h));
                                        if (scale < 1f) {
                                            decoder.setTargetSize(
                                                    Math.max(1, (int) (w * scale)),
                                                    Math.max(1, (int) (h * scale)));
                                        }
                                    });
                    pendingImage = bm;
                    runOnUiThread(() -> log("已选图片 " + bm.getWidth() + "x"
                            + bm.getHeight() + "，输入问题后点发送"));
                } catch (Exception e) {
                    runOnUiThread(() -> log("[ERR] 图片读取失败: " + e.getMessage()));
                }
            });
        }
    }

    /** 把 base_url / api_key / model / curl 示例整段复制到剪贴板（并写入日志便于 adb 核对）。 */
    private void copyConnInfo() {
        AiRuntime rt = AiRuntime.get();
        rt.install(this);
        String text = rt.connInfo();
        try {
            android.content.ClipboardManager cm = (android.content.ClipboardManager)
                    getSystemService(CLIPBOARD_SERVICE);
            if (cm != null) {
                cm.setPrimaryClip(android.content.ClipData.newPlainText("AI API", text));
                toast("URL 与 API Key 已复制到剪贴板");
            } else {
                toast("剪贴板不可用");
            }
        } catch (Throwable t) {
            toast("复制失败: " + t.getMessage());
        }
        AiDebug.i("apiinfo:\n" + text);
        log("已复制到剪贴板：");
        log(text);
    }

    /** 启/停本地 OpenAI 兼容服务端（默认常开；开关状态持久化，重启后仍生效）。 */
    private void toggleApiServer() {
        AiRuntime rt = AiRuntime.get();
        boolean on = !rt.isApiEnabled();
        rt.setApiEnabled(on);
        btnApi.setText(on ? "API:开" : "本地API");
        if (on) {
            log("[API] 已请求启动: http://127.0.0.1:" + AiRuntime.API_PORT + "/v1");
            log("[API] API_KEY=" + rt.apiKey());
            log("[API] 模型: " + AiHttpServer.MODEL_ID + "（OpenAI 兼容）");
        } else {
            log("[API] 已请求停止: tcp://" + AiRuntime.API_PORT);
        }
    }

    /** 外置模型放在 /sdcard 时需要"所有文件访问"权限（llama mmap 直读路径）。 */
    private void requestAllFilesAccess() {
        try {
            android.content.Intent i = new android.content.Intent(
                    android.provider.Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                    android.net.Uri.parse("package:" + getPackageName()));
            startActivity(i);
            log("请授予'所有文件访问'权限后点'重载模型'");
        } catch (Exception e) {
            log("[ERR] 无法打开权限页: " + e.getMessage());
        }
    }

    private void startRecording() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.RECORD_AUDIO}, REQ_RECORD_AUDIO);
            return;
        }
        executor.execute(() -> {
            int minBuf = AudioRecord.getMinBufferSize(AsrEngine.SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT);
            AudioRecord r = new AudioRecord(MediaRecorder.AudioSource.MIC,
                    AsrEngine.SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT,
                    Math.max(minBuf, 4 * 4096));
            try {
                r.startRecording();
            } catch (IllegalStateException e) {
                runOnUiThread(() -> log("[ERR] 录音启动失败: " + e.getMessage()));
                r.release();
                return;
            }
            recorder = r;
            recording = true;
            runOnUiThread(() -> log("（录音中…松开结束）"));

            // 流式识别：持续读取，每凑满约 0.8s 新音频即转写一次实时上屏
            java.util.List<float[]> parts = new java.util.ArrayList<>();
            int total = 0;
            int processed = 0;
            StringBuilder liveText = new StringBuilder();
            float[] buf = new float[1600];
            boolean streamingOk = true;
            while (recording) {
                int n = r.read(buf, 0, buf.length, AudioRecord.READ_BLOCKING);
                if (n <= 0) continue;
                parts.add(java.util.Arrays.copyOf(buf, n));
                total += n;
                if (total - processed >= AsrEngine.SAMPLE_RATE * 4 / 5) {
                    // 重叠窗口：多带 0.3s 上文音频，减少段边界截断词（文本侧再按 mergeTranscript 去重）
                    int from = Math.max(0, processed - OVERLAP_SAMPLES);
                    float[] tail = new float[total - from];
                    int off = 0;
                    for (float[] p : parts) {
                        int start = Math.max(from, off);
                        int end = Math.min(total, off + p.length);
                        if (end > start) {
                            System.arraycopy(p, start - off, tail, start - from,
                                    end - start);
                        }
                        off += p.length;
                    }
                    long t0 = android.os.SystemClock.elapsedRealtime();
                    String piece = asr.transcribe(tail);
                    long ms = android.os.SystemClock.elapsedRealtime() - t0;
                    float chunkSec = (total - processed) / (float) AsrEngine.SAMPLE_RATE;
                    float rtf = ms / 1000f / chunkSec;
                    if (rtf > 1.5f && streamingOk) {
                        streamingOk = false;
                        final float frtf = rtf;
                        runOnUiThread(() -> log(String.format(
                                "[WARN] 设备算力不足以实时流式（RTF=%.1f>1.0），"
                                        + "已降级为松手后整段识别", frtf)));
                    }
                    processed = total;
                    if (!piece.isEmpty()) {
                        String merged = mergeTranscript(liveText.toString(), piece);
                        liveText.setLength(0);
                        liveText.append(merged);
                        final String shown = merged;
                        runOnUiThread(() -> log("ASR(实时): " + shown));
                    }
                }
            }
            // 收尾：排空剩余缓冲后停麦
            while (r.read(buf, 0, buf.length, AudioRecord.READ_NON_BLOCKING) > 0) {
                parts.add(java.util.Arrays.copyOf(buf, buf.length));
                total += buf.length;
            }
            recorder = null;
            try {
                r.stop();
            } catch (IllegalStateException ignored) {
            }
            r.release();
            final int totalSamples = total;
            runOnUiThread(() -> log(String.format("（录音结束 %.1fs）",
                    totalSamples / (float) AsrEngine.SAMPLE_RATE)));

            if (total < AsrEngine.SAMPLE_RATE / 2) {
                runOnUiThread(() -> log("（录音太短，按住至少 1 秒）"));
                return;
            }
            if (!streamingOk) {
                AiDebug.i("asr streaming disabled: RTF>1.0 on this device (compute bound)");
            }
            // 终稿：全量音频重新转写（比分段拼接更准）
            float[] pcm = new float[total];
            int off = 0;
            for (float[] p : parts) {
                System.arraycopy(p, 0, pcm, off, p.length);
                off += p.length;
            }
            String text = asr.transcribe(pcm);
            if (!text.isEmpty()) transcript.append(text).append('\n');
            String shown = text.isEmpty() ? liveText.toString() : text;
            final String finalShown = shown.isEmpty() ? "（未识别到语音）" : shown;
            runOnUiThread(() -> append("ASR: " + finalShown));
        });
    }

    private void stopRecordingAndTranscribe() {
        // 只翻标志：采集循环读到标志后退出的逻辑在 startRecording 任务里
        recording = false;
    }

    private void runAsync(Runnable r) {
        runOnUiThread(() -> btnSend.setEnabled(false));
        executor.execute(() -> {
            try {
                r.run();
            } catch (Exception e) {
                runOnUiThread(() -> log("[ERR] " + e.getMessage()));
            } finally {
                runOnUiThread(() -> {
                    btnSend.setEnabled(true);
                    btnCorrect.setEnabled(true);
                    btnSummarize.setEnabled(true);
                });
            }
        });
    }

    /** 把 LLM 输出过滤思考区后追加显示，并收集进转写文本。 */
    private void appendVisible(String text) {
        ThinkFilter filter = new ThinkFilter();
        StringBuilder visible = new StringBuilder();
        for (int i = 0; i < text.length(); i += 64) {
            visible.append(filter.push(text.substring(i, Math.min(text.length(), i + 64))));
        }
        visible.append(filter.flush());
        runOnUiThread(() -> append(visible.toString()));
    }

    private void append(String line) {
        transcript.append(line).append('\n');
        log(line);
    }

    private void log(String line) {
        logView.append(line + "\n");
    }

    private void toast(String msg) {
        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show();
    }

    /**
     * 合并流式转写片段：相邻片段因音频有重叠窗口会产生重复文本，
     * 用「已显示文本的后缀」与「新片段的前缀」做最长公共匹配（上限 8 字）去重；
     * 匹配不到则直接拼接（安全回退，不会丢字）。
     */
    private static String mergeTranscript(String prev, String next) {
        if (prev == null || prev.isEmpty()) return next == null ? "" : next;
        if (next == null || next.isEmpty()) return prev;
        int max = Math.min(8, Math.min(prev.length(), next.length()));
        for (int k = max; k >= 1; k--) {
            if (prev.regionMatches(prev.length() - k, next, 0, k)) {
                return prev + next.substring(k);
            }
        }
        return prev + next;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQ_RECORD_AUDIO
                && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startRecording();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        recording = false;
        executor.shutdownNow();
        // 模型由常驻运行时（AiRuntime / AiBootService）持有，此处**不释放**，
        // 否则会打断后台本地 API 服务与已加载模型。
    }
}
