package com.ar.glass.sop;

import android.app.AlertDialog;
import android.content.ContentResolver;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Rect;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.ActivityResult;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;
import androidx.core.content.FileProvider;

import com.ar.glass.R;
import com.ar.glass.ui.GalleryActivity;
import com.ar.glass.vision.MarkedPointDetectorHolder;
import com.ar.glass.vision.MeterReading;
import com.ar.glass.vision.Vision;
import com.ar.glass.vision.realtime.WitnessStateEstimate;
import com.ar.glass.vision.realtime.WitnessTriage;
import com.ar.glass.vision.ui.BoxOverlay;
import com.ar.glass.vision.ui.WitnessReviewCrop;
import com.ar.glass.vision.ui.WitnessReviewOverlay;
import com.google.gson.JsonObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
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
    private View evidencePanel;
    private ImageView evidencePreview;
    private BoxOverlay evidenceOverlay;
    private Button captureButton;
    private Button phoneGalleryButton;
    private Button originalGalleryButton;
    private View galleryEvidenceSources;
    private TextView analysisText;
    private View witnessReviewPanel;
    private TextView witnessReviewProgress;
    private ImageView witnessReviewImage;
    private WitnessReviewOverlay witnessReviewOverlay;
    private TextView witnessAutomaticResult;
    private Spinner witnessPointDecision;
    private Button witnessPrevious;
    private Button witnessNext;
    private Button witnessFullImage;
    private Spinner decision;
    private EditText note;
    private Button saveStepButton;
    private Button submitButton;

    private SopModels.User currentUser;
    private SopModels.Assignment assignment;
    private String runId;
    private int stepIndex;
    private File evidenceFile;
    private File pendingCameraFile;
    private String evidenceSource = "";
    private String analysisSummary = "";
    private Map<String, Object> analysisValue = new HashMap<>();
    private final List<Map<String, Object>> pointResultMaps = new ArrayList<>();
    private List<MarkedPointDetectorHolder.Assessment> witnessAssessments = Collections.emptyList();
    private List<com.ar.glass.vision.YoloDetector.Detection> witnessDetections = Collections.emptyList();
    private WitnessReviewState witnessReviewState;
    private Bitmap evidenceBitmap;
    private Bitmap witnessReviewPatch;

    private ActivityResultLauncher<Uri> camera;
    private ActivityResultLauncher<String> phoneGallery;
    private ActivityResultLauncher<Intent> originalGallery;

    @Override protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_sop);
        api = new SopApiClient(this);
        bindViews();
        bindActions();
        camera = registerForActivityResult(new ActivityResultContracts.TakePicture(), success -> {
            File captured = pendingCameraFile;
            pendingCameraFile = null;
            if (success && captured != null) {
                decodeAndProcessEvidence(captured, "CAMERA", true);
            } else {
                if (captured != null && captured.exists()) captured.delete();
                setBusy(false, "未获取照片");
            }
        });
        phoneGallery = registerForActivityResult(
                new ActivityResultContracts.GetContent(), this::importPhoneEvidence);
        originalGallery = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(), this::importOriginalEvidence);
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
        evidencePanel = findViewById(R.id.panelSopEvidence);
        evidencePreview = findViewById(R.id.ivSopEvidence);
        evidenceOverlay = findViewById(R.id.overlaySopEvidence);
        captureButton = findViewById(R.id.btnCaptureStep);
        phoneGalleryButton = findViewById(R.id.btnSelectPhoneGallery);
        originalGalleryButton = findViewById(R.id.btnSelectOriginalGallery);
        galleryEvidenceSources = findViewById(R.id.layoutGalleryEvidenceSources);
        analysisText = findViewById(R.id.tvStepAnalysis);
        witnessReviewPanel = findViewById(R.id.panelWitnessPointReview);
        witnessReviewProgress = findViewById(R.id.tvWitnessReviewProgress);
        witnessReviewImage = findViewById(R.id.ivWitnessReviewCrop);
        witnessReviewOverlay = findViewById(R.id.overlayWitnessReviewRoi);
        witnessAutomaticResult = findViewById(R.id.tvWitnessAutomaticResult);
        witnessPointDecision = findViewById(R.id.spinnerWitnessPointDecision);
        witnessPrevious = findViewById(R.id.btnWitnessPrevious);
        witnessNext = findViewById(R.id.btnWitnessNext);
        witnessFullImage = findViewById(R.id.btnWitnessFullImage);
        decision = findViewById(R.id.spinnerStepDecision);
        note = findViewById(R.id.etStepNote);
        saveStepButton = findViewById(R.id.btnSaveStep);
        submitButton = findViewById(R.id.btnSubmitRun);
        ArrayAdapter<String> decisions = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_dropdown_item,
                new String[]{"确认正常", "疑似异常", "无法判断"});
        decision.setAdapter(decisions);
        witnessPointDecision.setAdapter(new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_dropdown_item,
                new String[]{"请选择逐点结论", "确认正常", "疑似松动", "无法判断/重拍"}));
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
        phoneGalleryButton.setOnClickListener(v -> phoneGallery.launch("image/*"));
        originalGalleryButton.setOnClickListener(v -> selectOriginalGalleryEvidence());
        witnessPrevious.setOnClickListener(v -> moveWitnessReview(false));
        witnessNext.setOnClickListener(v -> moveWitnessReview(true));
        witnessFullImage.setOnClickListener(v -> showWitnessFullImage());
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
            evidenceBitmap = null;
            resetWitnessReviews();
            stepNumber.setText("全部步骤已完成");
            stepTitle.setText("请提交本次巡检");
            stepInstruction.setText("提交后将进入后台人工复核队列");
            stepContract.setText("证据完整性门已在提交时再次校验");
            captureButton.setVisibility(View.GONE);
            galleryEvidenceSources.setVisibility(View.GONE);
            saveStepButton.setVisibility(View.GONE);
            evidencePanel.setVisibility(View.GONE);
            evidenceOverlay.setVisibility(View.GONE);
            decision.setVisibility(View.GONE);
            note.setVisibility(View.GONE);
            analysisText.setText("");
            progress.setProgress(100);
            return;
        }
        evidenceFile = null;
        evidenceSource = "";
        analysisSummary = "";
        analysisValue = new HashMap<>();
        evidenceBitmap = null;
        evidencePreview.setImageDrawable(null);
        evidenceOverlay.clear();
        resetWitnessReviews();
        evidencePanel.setVisibility(View.VISIBLE);
        evidenceOverlay.setVisibility(View.VISIBLE);
        captureButton.setVisibility(View.VISIBLE);
        galleryEvidenceSources.setVisibility(View.VISIBLE);
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
            pendingCameraFile = new File(directory, "sop_" + System.currentTimeMillis() + ".jpg");
            Uri uri = FileProvider.getUriForFile(this, "com.ar.glass.fileprovider", pendingCameraFile);
            setBusy(true, "等待现场拍照…");
            camera.launch(uri);
        } catch (Exception error) {
            pendingCameraFile = null;
            setBusy(false, "无法启动相机：" + error.getMessage());
        }
    }

    private void selectOriginalGalleryEvidence() {
        Intent intent = new Intent(this, GalleryActivity.class);
        intent.putExtra(GalleryActivity.EXTRA_MODE, GalleryActivity.MODE_ORIGINAL);
        intent.putExtra(GalleryActivity.EXTRA_SELECT_IMAGE, true);
        originalGallery.launch(intent);
    }

    private void importPhoneEvidence(Uri uri) {
        if (uri == null) {
            setBusy(false, "已取消选择，当前证据保持不变");
            return;
        }
        ContentResolver resolver = getContentResolver();
        String mime = resolver.getType(uri);
        importEvidence(() -> resolver.openInputStream(uri), mime, "PHONE_GALLERY");
    }

    private void importOriginalEvidence(ActivityResult result) {
        if (result.getResultCode() != RESULT_OK || result.getData() == null) {
            setBusy(false, "已取消选择，当前证据保持不变");
            return;
        }
        String path = result.getData().getStringExtra(GalleryActivity.EXTRA_SELECTED_IMAGE_PATH);
        File source = path == null ? null : new File(path);
        if (source == null || !source.isFile()) {
            setBusy(false, "原图库图片不存在，当前证据保持不变");
            return;
        }
        try {
            importEvidence(() -> new FileInputStream(source), SopEvidenceFiles.mediaType(source),
                    "APP_GALLERY");
        } catch (IllegalArgumentException error) {
            setBusy(false, error.getMessage() + "，当前证据保持不变");
        }
    }

    private void importEvidence(EvidenceInput input, String mime, String source) {
        setBusy(true, "正在读取图库图片…");
        analyzerExecutor.execute(() -> {
            File snapshot = null;
            try {
                snapshot = SopEvidenceFiles.copy(
                        input.open(), mime, evidenceDirectory(), System.currentTimeMillis());
                Bitmap bitmap = SopImageDecoder.decode(snapshot, 2200);
                File finalSnapshot = snapshot;
                runOnUiThread(() -> processEvidenceFile(finalSnapshot, bitmap, source));
            } catch (Exception error) {
                if (snapshot != null && snapshot.exists()) snapshot.delete();
                String message = error.getMessage() == null ? "图片读取失败" : error.getMessage();
                runOnUiThread(() -> setBusy(false, message + "，当前证据保持不变"));
            }
        });
    }

    private void decodeAndProcessEvidence(File file, String source, boolean deleteOnFailure) {
        analyzerExecutor.execute(() -> {
            try {
                Bitmap bitmap = SopImageDecoder.decode(file, 2200);
                runOnUiThread(() -> processEvidenceFile(file, bitmap, source));
            } catch (IOException error) {
                if (deleteOnFailure && file.exists()) file.delete();
                runOnUiThread(() -> setBusy(false, "照片读取失败，当前证据保持不变"));
            }
        });
    }

    private void processEvidenceFile(File file, Bitmap bitmap, String source) {
        evidenceOverlay.clear();
        resetWitnessReviews();
        saveStepButton.setEnabled(false);
        evidenceFile = file;
        evidenceSource = source;
        evidenceBitmap = bitmap;
        evidencePreview.setImageBitmap(bitmap);
        analyzeCurrentStep(bitmap);
    }

    private File evidenceDirectory() throws IOException {
        File directory = getExternalFilesDir("sop_evidence");
        if (directory == null) throw new IOException("外部存储不可用");
        if (!directory.exists() && !directory.mkdirs()) throw new IOException("无法创建证据目录");
        return directory;
    }

    private interface EvidenceInput {
        InputStream open() throws IOException;
    }

    private void analyzeCurrentStep(Bitmap bitmap) {
        SopModels.Step step = currentStep();
        if (step == null || bitmap == null) return;
        setBusy(true, "正在执行 " + step.title + "…");
        analyzerExecutor.execute(() -> {
            Map<String, Object> value = new HashMap<>();
            value.put("evidence_source", evidenceSource);
            String summary;
            int recommendedDecision = -1;
            List<com.ar.glass.vision.YoloDetector.Detection> overlayDetections = null;
            List<MarkedPointDetectorHolder.Assessment> pointAssessments = null;
            List<Map<String, Object>> pointResults = null;
            try {
                switch (step.type) {
                    case "QR":
                        String qr = Vision.get().decodeQrCode(bitmap);
                        value.put("qr_text", qr == null ? "" : qr);
                        summary = qr == null ? "未识别到二维码，请重拍或选择无法判断" : "二维码：" + qr;
                        break;
                    case "FASTENER_MARK":
                        MarkedPointDetectorHolder.Result detection = MarkedPointDetectorHolder.detect(this, bitmap);
                        overlayDetections = detection.detections;
                        pointAssessments = detection.assessments;
                        value.put("marked_point_count", detection.detections.size());
                        value.put("latency_ms", Math.round(detection.latencyMillis));
                        value.put("detector_latency_ms", Math.round(detection.detectorLatencyMillis));
                        value.put("state_latency_ms", Math.round(detection.stateLatencyMillis));
                        value.put("analyzer_version", "marked-point-v1+witness-roi-v1-assistive");
                        pointResults = new ArrayList<>();
                        int aligned = 0;
                        int suspected = 0;
                        int highSuspicion = 0;
                        int insufficient = 0;
                        for (MarkedPointDetectorHolder.Assessment assessment : detection.assessments) {
                            WitnessStateEstimate estimate = assessment.estimate;
                            Map<String, Object> point = new HashMap<>();
                            point.put("index", assessment.index);
                            point.put("bbox_xyxy", java.util.Arrays.asList(
                                    assessment.left, assessment.top, assessment.right, assessment.bottom));
                            point.put("detection_confidence", assessment.detectionConfidence);
                            point.put("triage", estimate.getTriage().name());
                            point.put("reason", estimate.getReason());
                            if (estimate.isMeasured()) {
                                point.put("angle_degrees", estimate.getAngleDegrees());
                                point.put("angle_lower_degrees", estimate.getLowerDegrees());
                                point.put("angle_upper_degrees", estimate.getUpperDegrees());
                                point.put("state_inference_ms", estimate.getInferenceMillis());
                            }
                            point.put("requires_human_confirmation", true);
                            pointResults.add(point);
                            if (estimate.getTriage() == WitnessTriage.LIKELY_ALIGNED) aligned++;
                            else if (estimate.getTriage() == WitnessTriage.POSSIBLE_DISPLACED) suspected++;
                            else if (estimate.getTriage() == WitnessTriage.HIGH_SUSPICION) highSuspicion++;
                            else insufficient++;
                        }
                        value.put("point_results", pointResults);
                        if (highSuspicion > 0 || suspected > 0) {
                            value.put("ai_triage", "REVIEW_REQUIRED");
                            recommendedDecision = 1;
                        } else if (!detection.assessments.isEmpty() && insufficient == 0) {
                            value.put("ai_triage", "LIKELY_ALIGNED");
                            recommendedDecision = 0;
                        } else {
                            value.put("ai_triage", "INSUFFICIENT");
                            recommendedDecision = 2;
                        }
                        value.put("requires_human_confirmation", true);
                        summary = "检出 " + detection.detections.size() + " 个防松检查点：正常倾向 "
                                + aligned + "，疑似错位 " + suspected + "，高疑似 "
                                + highSuspicion + "，需近拍 " + insufficient + "；请逐点人工确认";
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
            int finalRecommendedDecision = recommendedDecision;
            List<com.ar.glass.vision.YoloDetector.Detection> finalOverlayDetections = overlayDetections;
            List<MarkedPointDetectorHolder.Assessment> finalPointAssessments = pointAssessments;
            List<Map<String, Object>> finalPointResults = pointResults;
            runOnUiThread(() -> {
                analysisValue = value;
                analysisSummary = finalSummary;
                analysisText.setText(finalSummary);
                if (finalOverlayDetections == null) {
                    evidenceOverlay.clear();
                } else {
                    evidenceOverlay.setResults(
                            finalOverlayDetections, bitmap.getWidth(), bitmap.getHeight());
                }
                if (finalRecommendedDecision >= 0) {
                    decision.setSelection(finalRecommendedDecision);
                }
                renderPointReviews(
                        bitmap, finalOverlayDetections, finalPointAssessments, finalPointResults);
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
        if ("FASTENER_MARK".equals(step.type) && witnessReviewState != null) {
            witnessReviewState.setCurrentDecision(witnessPointDecision.getSelectedItemPosition());
            if (!witnessReviewState.allReviewed()) {
                setStatus("请先确认每一个防松检查点");
                return;
            }
            boolean anySuspected = false;
            boolean anyUnable = false;
            for (int index = 0; index < witnessReviewState.size(); index++) {
                int pointSelection = witnessReviewState.getDecision(index);
                String pointDecision;
                if (pointSelection == 1) pointDecision = "confirmed_aligned";
                else if (pointSelection == 2) {
                    pointDecision = "suspected_displaced";
                    anySuspected = true;
                } else {
                    pointDecision = "unable_to_judge";
                    anyUnable = true;
                }
                pointResultMaps.get(index).put("human_decision", pointDecision);
            }
            selected = anySuspected ? 1 : (anyUnable ? 2 : 0);
            decision.setSelection(selected);
        }
        String humanDecision = SopStepPolicy.humanDecision(selected);
        Map<String, Object> value = new HashMap<>(analysisValue);
        if ("FASTENER_MARK".equals(step.type)) {
            value.put("state", selected == 0
                    ? "ALIGNED" : (selected == 1 ? "SUSPECTED" : "INSUFFICIENT"));
            value.put("human_review_complete", true);
        }
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

    private void renderPointReviews(
            Bitmap bitmap,
            List<com.ar.glass.vision.YoloDetector.Detection> detections,
            List<MarkedPointDetectorHolder.Assessment> assessments,
            List<Map<String, Object>> resultMaps) {
        resetWitnessReviews();
        if (bitmap == null || detections == null || assessments == null || resultMaps == null
                || assessments.isEmpty() || detections.size() != assessments.size()
                || assessments.size() != resultMaps.size()) {
            return;
        }
        evidenceBitmap = bitmap;
        witnessAssessments = new ArrayList<>(assessments);
        witnessDetections = new ArrayList<>(detections);
        pointResultMaps.addAll(resultMaps);
        witnessReviewState = new WitnessReviewState(assessments.size());
        witnessReviewPanel.setVisibility(View.VISIBLE);
        renderWitnessReviewPoint();
    }

    private void resetWitnessReviews() {
        witnessReviewPanel.setVisibility(View.GONE);
        witnessReviewImage.setImageDrawable(null);
        witnessReviewOverlay.clear();
        witnessAssessments = Collections.emptyList();
        witnessDetections = Collections.emptyList();
        witnessReviewState = null;
        pointResultMaps.clear();
        if (witnessReviewPatch != null && !witnessReviewPatch.isRecycled()) {
            witnessReviewPatch.recycle();
        }
        witnessReviewPatch = null;
    }

    private void moveWitnessReview(boolean forward) {
        if (witnessReviewState == null) return;
        witnessReviewState.setCurrentDecision(witnessPointDecision.getSelectedItemPosition());
        boolean moved = forward
                ? witnessReviewState.moveNext() : witnessReviewState.movePrevious();
        if (moved) renderWitnessReviewPoint();
    }

    private void renderWitnessReviewPoint() {
        if (witnessReviewState == null || evidenceBitmap == null) return;
        int position = witnessReviewState.getCurrentIndex();
        MarkedPointDetectorHolder.Assessment assessment = witnessAssessments.get(position);
        WitnessReviewCrop crop = WitnessReviewCrop.fromNormalized(
                assessment.left,
                assessment.top,
                assessment.right,
                assessment.bottom,
                evidenceBitmap.getWidth(),
                evidenceBitmap.getHeight());
        Bitmap patch = renderSourceCropWithNeutralPadding(evidenceBitmap, crop);
        witnessReviewImage.setImageBitmap(patch);
        if (witnessReviewPatch != null && witnessReviewPatch != patch
                && !witnessReviewPatch.isRecycled()) {
            witnessReviewPatch.recycle();
        }
        witnessReviewPatch = patch;
        witnessReviewOverlay.setReview(
                crop, assessment.index, assessment.estimate.getTriage().name());
        witnessReviewProgress.setText(
                "逐点审核 " + (position + 1) + "/" + witnessReviewState.size());
        String automatic = automaticWitnessResult(assessment.estimate);
        if (crop.requiresCloserCapture()) {
            automatic += " · 目标小于32像素，请靠近重拍";
            if (witnessReviewState.getCurrentDecision() == 0) {
                witnessReviewState.setCurrentDecision(3);
            }
        }
        witnessAutomaticResult.setText("AI辅助：" + automatic);
        witnessPointDecision.setSelection(witnessReviewState.getCurrentDecision());
        witnessPrevious.setEnabled(position > 0);
        witnessNext.setEnabled(position + 1 < witnessReviewState.size());
    }

    private String automaticWitnessResult(WitnessStateEstimate estimate) {
        if (estimate.getTriage() == WitnessTriage.LIKELY_ALIGNED) {
            return String.format(Locale.CHINA, "正常倾向 %.1f°", estimate.getAngleDegrees());
        }
        if (estimate.getTriage() == WitnessTriage.POSSIBLE_DISPLACED) {
            return String.format(Locale.CHINA, "疑似错位 %.1f°", estimate.getAngleDegrees());
        }
        if (estimate.getTriage() == WitnessTriage.HIGH_SUSPICION) {
            return String.format(Locale.CHINA, "高疑似松动 %.1f°（待确认）", estimate.getAngleDegrees());
        }
        return "无法自动测量，请近拍";
    }

    private Bitmap renderSourceCropWithNeutralPadding(Bitmap source, WitnessReviewCrop crop) {
        int outputSize = Math.min(720, Math.max(1, Math.round(crop.getSide())));
        float scale = outputSize / crop.getSide();
        Bitmap output = Bitmap.createBitmap(
                outputSize, outputSize, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(output);
        canvas.drawColor(Color.rgb(112, 112, 112));
        float requestedLeft = crop.getRequestedLeft();
        float requestedTop = crop.getRequestedTop();
        Rect sourceRect = new Rect(
                Math.max(0, (int) Math.floor(requestedLeft)),
                Math.max(0, (int) Math.floor(requestedTop)),
                Math.min(source.getWidth(), (int) Math.ceil(requestedLeft + crop.getSide())),
                Math.min(source.getHeight(), (int) Math.ceil(requestedTop + crop.getSide())));
        if (!sourceRect.isEmpty()) {
            RectF destination = new RectF(
                    (sourceRect.left - requestedLeft) * scale,
                    (sourceRect.top - requestedTop) * scale,
                    (sourceRect.right - requestedLeft) * scale,
                    (sourceRect.bottom - requestedTop) * scale);
            canvas.drawBitmap(source, sourceRect, destination, null);
        }
        return output;
    }

    private void showWitnessFullImage() {
        if (witnessReviewState == null || evidenceBitmap == null) return;
        int position = witnessReviewState.getCurrentIndex();
        FrameLayout frame = new FrameLayout(this);
        ImageView image = new ImageView(this);
        image.setScaleType(ImageView.ScaleType.FIT_CENTER);
        image.setImageBitmap(evidenceBitmap);
        BoxOverlay overlay = new BoxOverlay(this);
        overlay.setResults(
                Collections.singletonList(witnessDetections.get(position)),
                evidenceBitmap.getWidth(),
                evidenceBitmap.getHeight());
        FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(420));
        frame.addView(image, params);
        frame.addView(overlay, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(420)));
        new AlertDialog.Builder(this)
                .setTitle("检查点 " + witnessAssessments.get(position).index + " · 原图定位")
                .setView(frame)
                .setPositiveButton("返回审核", null)
                .show();
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
        phoneGalleryButton.setEnabled(!busy);
        originalGalleryButton.setEnabled(!busy);
        saveStepButton.setEnabled(!busy && (currentStep() == null || !currentStep().requireEvidence || evidenceFile != null));
        submitButton.setEnabled(!busy);
        setStatus(message);
    }

    private void setStatus(String message) {
        status.setText(message == null ? "" : message);
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
        resetWitnessReviews();
        analyzerExecutor.shutdownNow();
        super.onDestroy();
    }
}
