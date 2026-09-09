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

    private TextView logView;
    private EditText input, llmPathInput, asrPathInput;
    private Button btnSend, btnTalk, btnCorrect, btnSummarize, btnClear, btnReload, btnImage;
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

        llm = new LlmEngine();
        asr = new AsrEngine();
        pipeline = new AiPipeline(llm);

        log("引擎初始化中（首次加载 LLM 约需 30-60s）…");
        executor.execute(() -> {
            LlmEngine.initGpuCompat(this);
            reloadModels();
        });

        // 模型接口：LLM 填 GGUF 文件路径、ASR 填 SenseVoice onnx 路径（tokens.txt 同目录），
        // 留空则使用内置模型；修改后点击"重载模型"生效
        btnReload.setOnClickListener(v -> {
            log("重载模型中…");
            executor.execute(this::reloadModels);
        });

        btnSend.setOnClickListener(v -> {
            String text = input.getText().toString().trim();
            android.graphics.Bitmap img = pendingImage;
            if (text.isEmpty() && img == null) return;
            input.setText("");
            pendingImage = null;
            final String prompt = text.isEmpty() ? "描述这张图片" : text;
            runAsync(() -> {
                if (img != null) {
                    append("我: [图片] " + prompt);
                    // 懒加载视觉投影器（一次性），随后走 VLM 图文通路
                    String mmproj = vlmPathFromAssets();
                    if (!llm.initVision(mmproj)) {
                        appendVisible("AI: [视觉编码器加载失败]");
                        return;
                    }
                    String reply = llm.generateWithImage(img, prompt, 512, 0.7f);
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

    /** 按当前输入框的路径重载 LLM/ASR（在后台线程调用）。路径留空用内置模型。 */
    private void reloadModels() {
        AiDebug.deviceSnapshot();
        AiDebug.i("reload: llmPath=" + llmPathInput.getText().toString().trim()
                + " (空=内置), asrPath=" + asrPathInput.getText().toString().trim()
                + " (空=内置)");
        try {
            String llmPath = llmPathInput.getText().toString().trim();
            if (llmPath.isEmpty()) {
                llmPath = LlmEngine.ensureModelFile(this, null);
            }
            final String llmPathFinal = llmPath;
            if (!new java.io.File(llmPathFinal).exists()) {
                runOnUiThread(() -> log("[ERR] LLM 路径不存在: " + llmPathFinal));
                return;
            }
            llm.load(llmPathFinal);

            String asrPath = asrPathInput.getText().toString().trim();
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
            recorder = new AudioRecord(MediaRecorder.AudioSource.MIC, AsrEngine.SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT,
                    Math.max(minBuf, 4 * 4096));
            recorder.startRecording();
            recording = true;
            runOnUiThread(() -> log("（录音中…松开结束）"));
        });
    }

    private void stopRecordingAndTranscribe() {
        executor.execute(() -> {
            recording = false;
            AudioRecord r = recorder;
            if (r == null) return;
            java.util.List<float[]> parts = new java.util.ArrayList<>();
            int total = 0;
            float[] buf = new float[1600];
            while (r.read(buf, 0, buf.length, AudioRecord.READ_BLOCKING) > 0) {
                parts.add(buf.clone());
                total += buf.length;
                if (!recording) break;
            }
            recorder = null;
            try {
                r.stop();
            } catch (IllegalStateException ignored) {
            }
            r.release();

            float[] pcm = new float[total];
            int off = 0;
            for (float[] p : parts) {
                System.arraycopy(p, 0, pcm, off, p.length);
                off += p.length;
            }
            if (total < AsrEngine.SAMPLE_RATE / 2) {
                runOnUiThread(() -> log("（录音太短）"));
                return;
            }
            String text = asr.transcribe(pcm);
            if (!text.isEmpty()) transcript.append(text).append('\n');
            String shown = text.isEmpty() ? "（未识别到语音）" : text;
            runOnUiThread(() -> append("ASR: " + shown));
        });
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
        if (llm != null) llm.release();
        if (asr != null) asr.release();
    }
}
