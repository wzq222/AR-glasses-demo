package com.ar.glass.sop;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;
import androidx.core.content.FileProvider;

import com.ar.glass.R;
import com.ar.glass.vision.MarkedPointDetectorHolder;
import com.ar.glass.vision.MeterReading;
import com.ar.glass.vision.Vision;
import com.google.gson.JsonObject;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TimeZone;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class SopActivity extends AppCompatActivity {
    private SopApiClient api;
    private final ExecutorService analyzerExecutor = Executors.newSingleThreadExecutor();

    private View loginPanel;
    private View taskPanel;
    private View executionPanel;
    private EditText username;
    private EditText password;
    private TextView userText;
    private TextView taskCount;
    private LinearLayout taskList;
    private TextView status;
    private TextView executionAsset;
    private TextView executionSop;
    private ProgressBar progress;
    private TextView stepNumber;
    private TextView stepTitle;
    private TextView stepInstruction;
    private TextView stepContract;
    private ImageView evidencePreview;
    private Button captureButton;
    private TextView analysisText;
    private Spinner decision;
    private EditText note;
    private Button saveStepButton;
    private Button submitButton;

    private SopModels.User currentUser;
    private SopModels.Assignment assignment;
    private String runId;
    private int stepIndex;
    private File evidenceFile;
    private String analysisSummary = "";
    private Map<String, Object> analysisValue = new HashMap<>();

    private ActivityResultLauncher<Uri> camera;

    @Override protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_sop);
        api = new SopApiClient(this);
        bindViews();
        bindActions();
        camera = registerForActivityResult(new ActivityResultContracts.TakePicture(), success -> {
            if (success && evidenceFile != null) {
                Bitmap bitmap = decode(evidenceFile, 2200);
                evidencePreview.setImageBitmap(bitmap);
                analyzeCurrentStep(bitmap);
            } else {
                setBusy(false, "未获取照片");
            }
        });
        if (api.hasSession()) restoreSession(); else showLogin();
    }

    private void bindViews() {
        loginPanel = findViewById(R.id.panelSopLogin);
        taskPanel = findViewById(R.id.panelSopTasks);
        executionPanel = findViewById(R.id.panelSopExecution);
        username = findViewById(R.id.etSopUsername);
        password = findViewById(R.id.etSopPassword);
        userText = findViewById(R.id.tvSopUser);
        taskCount = findViewById(R.id.tvSopTaskCount);
        taskList = findViewById(R.id.taskList);
        status = findViewById(R.id.tvSopStatus);
        executionAsset = findViewById(R.id.tvExecutionAsset);
        executionSop = findViewById(R.id.tvExecutionSop);
        progress = findViewById(R.id.progressSop);
        stepNumber = findViewById(R.id.tvStepNumber);
        stepTitle = findViewById(R.id.tvStepTitle);
        stepInstruction = findViewById(R.id.tvStepInstruction);
        stepContract = findViewById(R.id.tvStepContract);
        evidencePreview = findViewById(R.id.ivSopEvidence);
        captureButton = findViewById(R.id.btnCaptureStep);
        analysisText = findViewById(R.id.tvStepAnalysis);
        decision = findViewById(R.id.spinnerStepDecision);
        note = findViewById(R.id.etStepNote);
        saveStepButton = findViewById(R.id.btnSaveStep);
        submitButton = findViewById(R.id.btnSubmitRun);
        ArrayAdapter<String> decisions = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_dropdown_item,
                new String[]{"确认正常", "疑似异常", "无法判断"});
        decision.setAdapter(decisions);
    }

    private void bindActions() {
        findViewById(R.id.btnSopBack).setOnClickListener(v -> finish());
        findViewById(R.id.btnSopLogin).setOnClickListener(v -> login());
        findViewById(R.id.btnSopRefresh).setOnClickListener(v -> loadAssignments());
        findViewById(R.id.btnSopLogout).setOnClickListener(v -> {
            api.clearSession();
            currentUser = null;
            showLogin();
        });
        findViewById(R.id.btnBackTasks).setOnClickListener(v -> showTasks());
        captureButton.setOnClickListener(v -> captureEvidence());
        saveStepButton.setOnClickListener(v -> saveCurrentStep());
        submitButton.setOnClickListener(v -> submitRun());
    }

    private void showLogin() {
        loginPanel.setVisibility(View.VISIBLE);
        taskPanel.setVisibility(View.GONE);
        executionPanel.setVisibility(View.GONE);
        status.setText("");
    }

    private void restoreSession() {
        setStatus("正在恢复登录状态…");
        api.me(new SopApiClient.Result<SopModels.User>() {
            @Override public void onSuccess(SopModels.User value) {
                currentUser = value;
                showTasks();
            }
            @Override public void onError(String message) {
                api.clearSession();
                showLogin();
                setStatus(message);
            }
        });
    }

    private void login() {
        String name = username.getText().toString().trim();
        String secret = password.getText().toString();
        if (name.isEmpty() || secret.length() < 8) {
            setStatus("请输入巡检账号和密码");
            return;
        }
        setStatus("正在登录…");
        api.login(name, secret, new SopApiClient.Result<SopModels.LoginResponse>() {
            @Override public void onSuccess(SopModels.LoginResponse value) {
                currentUser = value.user;
                password.setText("");
                showTasks();
            }
            @Override public void onError(String message) { setStatus(message); }
        });
    }

    private void showTasks() {
        loginPanel.setVisibility(View.GONE);
        executionPanel.setVisibility(View.GONE);
        taskPanel.setVisibility(View.VISIBLE);
        userText.setText(currentUser == null ? "巡检账号" : currentUser.displayName);
        loadAssignments();
    }

    private void loadAssignments() {
        setStatus("正在同步任务…");
        api.assignments(new SopApiClient.Result<List<SopModels.Assignment>>() {
            @Override public void onSuccess(List<SopModels.Assignment> values) {
                List<SopModels.Assignment> active = new ArrayList<>();
                for (SopModels.Assignment item : values) {
                    if (!"completed".equals(item.status) && !"cancelled".equals(item.status)) active.add(item);
                }
                taskCount.setText("待处理 " + active.size() + " 项");
                renderTasks(active);
                setStatus(active.isEmpty() ? "当前没有待处理任务" : "任务已同步");
            }
            @Override public void onError(String message) { setStatus(message); }
        });
    }

    private void renderTasks(List<SopModels.Assignment> assignments) {
        taskList.removeAllViews();
        for (SopModels.Assignment item : assignments) {
            CardView card = new CardView(this);
            LinearLayout.LayoutParams cardParams = new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
            cardParams.bottomMargin = dp(10);
            card.setLayoutParams(cardParams);
            card.setRadius(dp(8));
            LinearLayout content = new LinearLayout(this);
            content.setOrientation(LinearLayout.VERTICAL);
            content.setPadding(dp(15), dp(13), dp(15), dp(13));
            TextView asset = text(item.assetCode, 17, true, "#172733");
            TextView sop = text(item.title + " · v" + item.version + " · " + item.steps.size() + "步", 13, false, "#607483");
            sop.setPadding(0, dp(3), 0, dp(9));
            Button open = new Button(this);
            open.setText("in_progress".equals(item.status) ? "继续巡检" : "开始巡检");
            open.setTextColor(Color.WHITE);
            open.setBackgroundTintList(android.content.res.ColorStateList.valueOf(Color.parseColor("#176FA8")));
            open.setOnClickListener(v -> startAssignment(item));
            content.addView(asset);
            content.addView(sop);
            content.addView(open);
            card.addView(content);
            taskList.addView(card);
        }
    }

    private TextView text(String value, int size, boolean bold, String color) {
        TextView view = new TextView(this);
        view.setText(value == null ? "-" : value);
        view.setTextSize(size);
        view.setTextColor(Color.parseColor(color));
        if (bold) view.setTypeface(null, android.graphics.Typeface.BOLD);
        return view;
    }

    private void startAssignment(SopModels.Assignment selected) {
        setStatus("正在建立巡检会话…");
        api.startRun(selected.id, new SopApiClient.Result<SopModels.Run>() {
            @Override public void onSuccess(SopModels.Run run) {
                assignment = selected;
                runId = run.id;
                stepIndex = 0;
                taskPanel.setVisibility(View.GONE);
                executionPanel.setVisibility(View.VISIBLE);
                executionAsset.setText(selected.assetCode);
                executionSop.setText(selected.title + " · v" + selected.version);
                renderStep();
                setStatus("巡检会话已开始");
            }
            @Override public void onError(String message) { setStatus(message); }
        });
    }

    private SopModels.Step currentStep() {
        if (assignment == null || assignment.steps == null || stepIndex >= assignment.steps.size()) return null;
        return assignment.steps.get(stepIndex);
    }

    private void renderStep() {
        SopModels.Step step = currentStep();
        int total = assignment.steps.size();
        progress.setProgress(total == 0 ? 0 : Math.round(stepIndex * 100f / total));
        submitButton.setVisibility(step == null ? View.VISIBLE : View.GONE);
        if (step == null) {
            stepNumber.setText("全部步骤已完成");
            stepTitle.setText("请提交本次巡检");
            stepInstruction.setText("提交后将进入后台人工复核队列");
            stepContract.setText("证据完整性门已在提交时再次校验");
            captureButton.setVisibility(View.GONE);
            saveStepButton.setVisibility(View.GONE);
            evidencePreview.setVisibility(View.GONE);
            decision.setVisibility(View.GONE);
            note.setVisibility(View.GONE);
            analysisText.setText("");
            progress.setProgress(100);
            return;
        }
        evidenceFile = null;
        analysisSummary = "";
        analysisValue = new HashMap<>();
        evidencePreview.setImageDrawable(null);
        evidencePreview.setVisibility(View.VISIBLE);
        captureButton.setVisibility(View.VISIBLE);
        saveStepButton.setVisibility(View.VISIBLE);
        decision.setVisibility(View.VISIBLE);
        note.setVisibility(View.VISIBLE);
        note.setText("");
        decision.setSelection(0);
        stepNumber.setText("步骤 " + (stepIndex + 1) + "/" + total);
        stepTitle.setText(step.title);
        stepInstruction.setText(step.instruction);
        List<String> contract = new ArrayList<>();
        if (!"manual".equals(step.analyzer())) contract.add("自动检测");
        if (step.requireEvidence) contract.add("照片证据");
        if (step.requireHumanConfirmation) contract.add("人工确认");
        stepContract.setText(android.text.TextUtils.join(" · ", contract));
        captureButton.setText(step.requireEvidence ? "拍摄本步骤证据" : "拍照辅助判断（可选）");
        analysisText.setText("等待采集");
        saveStepButton.setEnabled(!step.requireEvidence);
    }

    private void captureEvidence() {
        try {
            File directory = getExternalFilesDir("sop_evidence");
            if (directory == null) throw new IllegalStateException("外部存储不可用");
            if (!directory.exists() && !directory.mkdirs()) throw new IllegalStateException("无法创建证据目录");
            evidenceFile = new File(directory, "sop_" + System.currentTimeMillis() + ".jpg");
            Uri uri = FileProvider.getUriForFile(this, "com.ar.glass.fileprovider", evidenceFile);
            setBusy(true, "等待现场拍照…");
            camera.launch(uri);
        } catch (Exception error) {
            setBusy(false, "无法启动相机：" + error.getMessage());
        }
    }

    private void analyzeCurrentStep(Bitmap bitmap) {
        SopModels.Step step = currentStep();
        if (step == null || bitmap == null) return;
        setBusy(true, "正在执行 " + step.title + "…");
        analyzerExecutor.execute(() -> {
            Map<String, Object> value = new HashMap<>();
            String summary;
            try {
                switch (step.type) {
                    case "QR":
                        String qr = Vision.get().decodeQrCode(bitmap);
                        value.put("qr_text", qr == null ? "" : qr);
                        summary = qr == null ? "未识别到二维码，请重拍或选择无法判断" : "二维码：" + qr;
                        break;
                    case "FASTENER_MARK":
                        MarkedPointDetectorHolder.Result detection = MarkedPointDetectorHolder.detect(this, bitmap);
                        value.put("marked_point_count", detection.detections.size());
                        value.put("latency_ms", Math.round(detection.latencyMillis));
                        value.put("state", "INSUFFICIENT");
                        summary = "检出 " + detection.detections.size() + " 个防松标记检查点；松动状态请人工确认";
                        break;
                    case "METER":
                        MeterReading reading = Vision.get().readMeter(bitmap);
                        String display = reading == null ? "" : reading.getDisplayText();
                        value.put("reading", display);
                        value.put("gear", reading == null ? "" : reading.gear);
                        summary = display.isEmpty() ? "未识别到稳定读数，请人工填写备注或重拍" : "识别读数：" + display;
                        break;
                    case "PHOTO":
                        value.put("captured", true);
                        summary = "现场照片已采集";
                        break;
                    default:
                        value.put("manual", true);
                        summary = "照片已采集，请人工确认结果";
                }
            } catch (Throwable error) {
                value.put("analysis_error", error.getClass().getSimpleName());
                summary = "自动检测未完成，请人工判断或重新采集";
            }
            String finalSummary = summary;
            runOnUiThread(() -> {
                analysisValue = value;
                analysisSummary = finalSummary;
                analysisText.setText(finalSummary);
                setBusy(false, "检测完成，请确认本步结果");
                saveStepButton.setEnabled(true);
            });
        });
    }

    private void saveCurrentStep() {
        SopModels.Step step = currentStep();
        if (step == null) return;
        if (step.requireEvidence && (evidenceFile == null || !evidenceFile.isFile())) {
            setStatus("本步骤必须先拍摄现场证据");
            return;
        }
        int selected = decision.getSelectedItemPosition();
        String humanDecision = SopStepPolicy.humanDecision(selected);
        Map<String, Object> value = new HashMap<>(analysisValue);
        value.put("analysis_summary", analysisSummary);
        value.put("operator_note", note.getText().toString().trim());
        value.put("decision_label", decision.getSelectedItem().toString());
        value.put("capture_source", "PHONE");
        SopModels.StepPayload payload = new SopModels.StepPayload();
        payload.idempotencyKey = SopStepPolicy.idempotencyKey(runId, step.key);
        payload.status = SopStepPolicy.statusForDecision(selected);
        payload.value = value;
        payload.confidence = null;
        payload.requiresHumanReview = step.requireHumanConfirmation;
        payload.humanDecision = step.requireHumanConfirmation ? humanDecision : null;
        payload.analyzerVersion = step.analyzer();
        payload.errorCode = null;
        payload.capturedAt = isoNow();
        setBusy(true, "正在保存步骤…");
        api.saveStep(runId, step.key, payload, new SopApiClient.Result<JsonObject>() {
            @Override public void onSuccess(JsonObject ignored) {
                if (step.requireEvidence) uploadEvidence(step); else finishStep();
            }
            @Override public void onError(String message) { setBusy(false, message); }
        });
    }

    private void uploadEvidence(SopModels.Step step) {
        setStatus("正在上传原始证据…");
        api.uploadEvidence(runId, step.key, evidenceFile, new SopApiClient.Result<JsonObject>() {
            @Override public void onSuccess(JsonObject ignored) { finishStep(); }
            @Override public void onError(String message) { setBusy(false, message); }
        });
    }

    private void finishStep() {
        setBusy(false, "步骤已保存，证据已上传");
        stepIndex++;
        renderStep();
    }

    private void submitRun() {
        setBusy(true, "正在提交整次巡检…");
        api.submitRun(runId, new SopApiClient.Result<JsonObject>() {
            @Override public void onSuccess(JsonObject ignored) {
                setBusy(false, "巡检已提交，等待后台复核");
                Toast.makeText(SopActivity.this, "巡检提交成功", Toast.LENGTH_LONG).show();
                showTasks();
            }
            @Override public void onError(String message) { setBusy(false, message); }
        });
    }

    private void setBusy(boolean busy, String message) {
        captureButton.setEnabled(!busy);
        saveStepButton.setEnabled(!busy && (currentStep() == null || !currentStep().requireEvidence || evidenceFile != null));
        submitButton.setEnabled(!busy);
        setStatus(message);
    }

    private void setStatus(String message) {
        status.setText(message == null ? "" : message);
    }

    private Bitmap decode(File file, int maxEdge) {
        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(file.getAbsolutePath(), bounds);
        int sample = 1;
        while (Math.max(bounds.outWidth, bounds.outHeight) / sample > maxEdge) sample *= 2;
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = sample;
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        return BitmapFactory.decodeFile(file.getAbsolutePath(), options);
    }

    private String isoNow() {
        SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US);
        format.setTimeZone(TimeZone.getTimeZone("UTC"));
        return format.format(new Date());
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    @Override protected void onDestroy() {
        analyzerExecutor.shutdownNow();
        super.onDestroy();
    }
}
