package com.ar.glass.vision;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

/**
 * YOLO 离线目标检测（ONNX Runtime，NNAPI 硬件加速，失败自动回退 CPU）。
 *
 * 与 glasses demo 的集成方式：
 *  - 模型从 assets/models/<model> 复制到应用私有目录后由 ONNX Runtime 加载
 *  - detect(Bitmap) 返回归一化坐标的检测框列表，供 UI 绘制 / TTS 播报
 *
 * 输入输出布局兼容：
 *  - 输出 [1, 4+nc, N]（ultralytics 默认）与 [1, N, 4+nc]（需转置）
 */
public final class YoloDetector {

    private static final String TAG = "YoloDetector";
    private static final String MODEL_ASSET = "models/screw_detect_v2_best.onnx";
    private static final String CACHED_MODEL = "yolo_model.onnx";
    private static final int INPUT_SIZE_DEFAULT = 640;

    /** 单个检测结果：坐标为相对原图的归一化值 [0,1] */
    public static class Detection {
        public final float x1, y1, x2, y2;
        public final float score;
        public final int classId;
        public final String className;
        /** Optional assistive witness-line result; coordinates are normalized to the full image. */
        public final String witnessTriage;
        public final float witnessAngleDegrees;
        public final float[] witnessPoints;

        Detection(float x1, float y1, float x2, float y2, float score, int classId, String className) {
            this(x1, y1, x2, y2, score, classId, className, null, Float.NaN, null);
        }

        Detection(
                float x1,
                float y1,
                float x2,
                float y2,
                float score,
                int classId,
                String className,
                String witnessTriage,
                float witnessAngleDegrees,
                float[] witnessPoints) {
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
            this.score = score;
            this.classId = classId;
            this.className = className;
            this.witnessTriage = witnessTriage;
            this.witnessAngleDegrees = witnessAngleDegrees;
            this.witnessPoints = witnessPoints == null ? null : witnessPoints.clone();
        }
    }

    private static volatile YoloDetector sInstance;

    public static YoloDetector get(Context context) {
        if (sInstance == null) {
            synchronized (YoloDetector.class) {
                if (sInstance == null) {
                    sInstance = new YoloDetector(context.getApplicationContext());
                }
            }
        }
        return sInstance;
    }

    private final OrtEnvironment mEnv;
    private OrtSession mSession;
    private final int mInputSize;
    private final List<String> mClassNames = new ArrayList<>();

    private float mConfThreshold = 0.25f;
    private float mIouThreshold = 0.45f;

    private YoloDetector(Context context) {
        mEnv = OrtEnvironment.getEnvironment();
        File cached = ensureModelFile(context);
        OrtSession session = null;
        int inputSize = INPUT_SIZE_DEFAULT;
        if (cached != null) {
            try {
                OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
                opts.setIntraOpNumThreads(4);
                try {
                    opts.addNnapi(); // GPU/NPU 加速；不支持时由我们回退 CPU
                    session = mEnv.createSession(cached.getAbsolutePath(), opts);
                    Log.i(TAG, "ONNX session created with NNAPI");
                } catch (Throwable gpuErr) {
                    Log.w(TAG, "NNAPI init failed, fallback CPU: " + gpuErr.getMessage());
                    opts.close();
                    opts = new OrtSession.SessionOptions();
                    opts.setIntraOpNumThreads(4);
                    session = mEnv.createSession(cached.getAbsolutePath(), opts);
                    Log.i(TAG, "ONNX session created on CPU");
                }
            } catch (Throwable e) {
                Log.e(TAG, "create session failed", e);
                session = null;
            }
        }
        mSession = session;
        if (session != null) {
            inputSize = readInputSize(session);
            readClassNames(session);
        }
        mInputSize = inputSize;
        Log.i(TAG, "YoloDetector ready: inputSize=" + mInputSize
                + " classes=" + mClassNames + " loaded=" + (session != null));
    }

    public boolean isReady() {
        return mSession != null;
    }

    public void setConfThreshold(float v) {
        mConfThreshold = v;
    }

    public void release() {
        if (mSession != null) {
            try { mSession.close(); } catch (Exception ignored) {}
            mSession = null;
        }
        sInstance = null;
    }

    /** 把 assets 里的模型复制到 filesDir（ONNX Runtime 需要文件路径或字节数组，这里用字节数组亦可，但缓存便于复用） */
    private File ensureModelFile(Context context) {
        try {
            File out = new File(context.getFilesDir(), CACHED_MODEL);
            if (out.exists() && out.length() > 0) return out;
            InputStream is = context.getAssets().open(MODEL_ASSET);
            FileOutputStream fos = new FileOutputStream(out);
            byte[] buf = new byte[8192];
            int n;
            while ((n = is.read(buf)) > 0) fos.write(buf, 0, n);
            fos.close();
            is.close();
            return out;
        } catch (Exception e) {
            Log.e(TAG, "copy model failed: " + e.getMessage());
            return null;
        }
    }

    private int readInputSize(OrtSession session) {
        try {
            TensorInfo info = (TensorInfo) session.getInputInfo().values().iterator().next().getInfo();
            long[] shape = info.getShape();
            if (shape != null && shape.length >= 4 && shape[2] > 0 && shape[3] > 0) {
                return (int) shape[2];
            }
        } catch (Exception ignored) {}
        return INPUT_SIZE_DEFAULT;
    }

    /** 读取 ultralytics 导出的 metadata names；读不到时用 class_0/1/... */
    private void readClassNames(OrtSession session) {
        mClassNames.clear();
        try {
            Map<String, String> meta = session.getMetadata().getCustomMetadata();
            String names = meta != null ? meta.get("names") : null;
            if (names != null && !names.isEmpty()) {
                // 形如 {0: 'screw', 1: 'nut'} 或 {0: screw, 1: nut}
                java.util.regex.Matcher m =
                        java.util.regex.Pattern.compile("(\\d+)\\s*:\\s*'?([^,'}\"']+)")
                                .matcher(names);
                List<int[]> idx = new ArrayList<>();
                List<String> nm = new ArrayList<>();
                while (m.find()) {
                    idx.add(new int[]{Integer.parseInt(m.group(1))});
                    nm.add(m.group(2).trim());
                }
                if (!nm.isEmpty()) {
                    int max = 0;
                    for (int[] i : idx) max = Math.max(max, i[0]);
                    String[] arr = new String[max + 1];
                    for (int i = 0; i < nm.size(); i++) arr[idx.get(i)[0]] = nm.get(i);
                    for (int i = 0; i < arr.length; i++) {
                        mClassNames.add(arr[i] != null ? arr[i] : ("class_" + i));
                    }
                }
            }
        } catch (Exception ignored) {}
        if (mClassNames.isEmpty()) mClassNames.add("target");
    }

    /**
     * 对一张图执行检测。
     * @return 检测框列表（归一化坐标）；引擎未就绪返回空列表
     */
    public List<Detection> detect(Bitmap src) {
        List<Detection> empty = Collections.emptyList();
        OrtSession session = mSession;
        if (session == null || src == null) return empty;

        int size = mInputSize;
        int frameW = src.getWidth();
        int frameH = src.getHeight();

        // letterbox
        float scale = Math.min(size / (float) frameW, size / (float) frameH);
        int newW = Math.max(1, Math.round(frameW * scale));
        int newH = Math.max(1, Math.round(frameH * scale));
        float padX = (size - newW) / 2f;
        float padY = (size - newH) / 2f;

        Bitmap input = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(input);
        canvas.drawColor(Color.rgb(114, 114, 114));
        Bitmap scaled = Bitmap.createScaledBitmap(src, newW, newH, true);
        canvas.drawBitmap(scaled, padX, padY, null);
        if (scaled != src) scaled.recycle();

        // HWC_ARGB → NCHW RGB /255
        int[] pixels = new int[size * size];
        input.getPixels(pixels, 0, size, 0, 0, size, size);
        input.recycle();
        int area = size * size;
        float[] data = new float[3 * area];
        for (int i = 0; i < area; i++) {
            int p = pixels[i];
            data[i] = ((p >> 16) & 0xFF) / 255f;
            data[area + i] = ((p >> 8) & 0xFF) / 255f;
            data[2 * area + i] = (p & 0xFF) / 255f;
        }

        try (OnnxTensor inTensor = OnnxTensor.createTensor(
                mEnv, FloatBuffer.wrap(data), new long[]{1, 3, size, size});
             OrtSession.Result outputs = session.run(
                     Collections.singletonMap(session.getInputNames().iterator().next(), inTensor))) {

            // 找最大输出张量（ultralytics 导出只有一个主输出）
            OnnxTensor outTensor = null;
            long best = -1;
            for (Map.Entry<String, ai.onnxruntime.OnnxValue> e : outputs) {
                if (e.getValue() instanceof OnnxTensor) {
                    OnnxTensor t = (OnnxTensor) e.getValue();
                    long n = 1;
                    for (long d : t.getInfo().getShape()) n *= Math.max(1, d);
                    if (n > best) { best = n; outTensor = t; }
                }
            }
            if (outTensor == null) return empty;

            long[] shape = outTensor.getInfo().getShape();
            float[] raw = outTensor.getFloatBuffer().array().clone();
            if (shape.length != 3) {
                Log.w(TAG, "unsupported output dims: " + java.util.Arrays.toString(shape));
                return empty;
            }
            int d1 = (int) shape[1];
            int d2 = (int) shape[2];

            // 统一为 [C=4+nc, N] 布局
            int channels;
            int anchors;
            float[] flat;
            if (d1 <= d2) {
                channels = d1;
                anchors = d2;
                flat = raw;
            } else {
                channels = d2;
                anchors = d1;
                flat = transpose(raw, d1, d2);
            }
            if (channels <= 4) return empty;

            return postprocess(flat, channels, anchors, scale, padX, padY, frameW, frameH);
        } catch (Exception e) {
            Log.e(TAG, "detect failed", e);
            return empty;
        }
    }

    private static float[] transpose(float[] src, int rows, int cols) {
        float[] out = new float[src.length];
        for (int r = 0; r < rows; r++) {
            int base = r * cols;
            for (int c = 0; c < cols; c++) {
                out[c * rows + r] = src[base + c];
            }
        }
        return out;
    }

    /** 解码 [C, N] 输出：阈值过滤 → 反 letterbox → 归一化 → NMS */
    private List<Detection> postprocess(float[] flat, int channels, int anchors,
                                        float scale, float padX, float padY,
                                        int frameW, int frameH) {
        int numClasses = channels - 4;
        List<Detection> cands = new ArrayList<>(64);
        for (int i = 0; i < anchors; i++) {
            float bestScore = 0f;
            int bestCls = 0;
            for (int c = 0; c < numClasses; c++) {
                float sc = flat[(4 + c) * anchors + i];
                if (sc > bestScore) { bestScore = sc; bestCls = c; }
            }
            if (bestScore < mConfThreshold) continue;

            float cx = flat[i];
            float cy = flat[anchors + i];
            float w = flat[2 * anchors + i];
            float h = flat[3 * anchors + i];

            // letterbox 像素 → 原图像素 → 归一化
            float px1 = (cx - w / 2f - padX) / scale / frameW;
            float py1 = (cy - h / 2f - padY) / scale / frameH;
            float px2 = (cx + w / 2f - padX) / scale / frameW;
            float py2 = (cy + h / 2f - padY) / scale / frameH;

            px1 = clamp01(px1); py1 = clamp01(py1);
            px2 = clamp01(px2); py2 = clamp01(py2);
            cands.add(new Detection(px1, py1, px2, py2, bestScore, bestCls,
                    mClassNames.get(Math.min(bestCls, mClassNames.size() - 1))));
        }
        return nms(cands);
    }

    private static float clamp01(float v) { return v < 0 ? 0 : (Math.min(v, 1f)); }

    /** 按类别的贪心 NMS */
    private static List<Detection> nms(List<Detection> in) {
        if (in.isEmpty()) return in;
        List<Detection> sorted = new ArrayList<>(in);
        Collections.sort(sorted, (a, b) -> Float.compare(b.score, a.score));
        List<Detection> keep = new ArrayList<>();
        boolean[] removed = new boolean[sorted.size()];
        for (int i = 0; i < sorted.size(); i++) {
            if (removed[i]) continue;
            Detection a = sorted.get(i);
            keep.add(a);
            for (int j = i + 1; j < sorted.size(); j++) {
                if (removed[j]) continue;
                Detection b = sorted.get(j);
                if (a.classId == b.classId && iou(a, b) > 0.45f) removed[j] = true;
            }
        }
        return keep;
    }

    private static float iou(Detection a, Detection b) {
        float ix1 = Math.max(a.x1, b.x1);
        float iy1 = Math.max(a.y1, b.y1);
        float ix2 = Math.min(a.x2, b.x2);
        float iy2 = Math.min(a.y2, b.y2);
        float iw = Math.max(0f, ix2 - ix1);
        float ih = Math.max(0f, iy2 - iy1);
        float inter = iw * ih;
        float union = (a.x2 - a.x1) * (a.y2 - a.y1)
                + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
        return union <= 0 ? 0 : inter / union;
    }
}
