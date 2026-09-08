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

    private TextView logView;
    private EditText input;
    private Button btnSend, btnTalk, btnCorrect, btnSummarize, btnClear;

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
        btnSend = findViewById(R.id.ai_send);
        btnTalk = findViewById(R.id.ai_talk);
        btnCorrect = findViewById(R.id.ai_correct);
        btnSummarize = findViewById(R.id.ai_summarize);
        btnClear = findViewById(R.id.ai_clear);

        llm = new LlmEngine();
        asr = new AsrEngine();
        pipeline = new AiPipeline(llm);

        log("引擎初始化中（首次加载 LLM 约需 30-60s）…");
        executor.execute(() -> {
            try {
                String modelPath = LlmEngine.ensureModelFile(this, null);
                llm.load(modelPath);
                asr.load(this);
                runOnUiThread(() -> log("[OK] 引擎就绪 GPU=" + llm.nativeIsGpuActive()));
            } catch (Exception e) {
                runOnUiThread(() -> log("[ERR] 初始化失败: " + e.getMessage()));
            }
        });

        btnSend.setOnClickListener(v -> {
            String text = input.getText().toString().trim();
            if (text.isEmpty()) return;
            input.setText("");
            runAsync(() -> {
                append("我: " + text);
                String reply = pipeline.chat(text);
                appendVisible("AI: " + reply);
            });
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
