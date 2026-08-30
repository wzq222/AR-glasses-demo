package com.ar.glass.voice;

import android.content.Context;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.media.ToneGenerator;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.speech.tts.TextToSpeech;
import android.util.Log;

import com.iflytek.sparkchain.core.asr.ASR;
import com.iflytek.sparkchain.core.asr.AsrCallbacks;

import java.io.ByteArrayOutputStream;
import java.util.Arrays;
import java.util.Locale;

/**
 * 语音控制与播报：
 * - 采集：按住说话时，通过蓝牙 SCO 采集眼镜麦克风（AudioRecord，16k/16bit/mono PCM）
 * - 识别：松开后调用讯飞 SparkChain 在线「语音听写」（ASR）做云端识别
 * - 控制：识别结果包含「拍照」时回调 {@link Listener#onKeywordDetected}
 * - 播报：通过 TTS 把文本播放到眼镜扬声器（STREAM_VOICE_CALL + SCO 路由）
 */
public class VoiceController {

    public interface Listener {
        void onKeywordDetected(String keyword);
        void onSpeechText(String text);
        void onListeningChanged(boolean listening);
        void onError(String message);
    }

    private static final String TAG = "VoiceController";

    private static final int SAMPLE_RATE = 16000;

    private static final String[] KEYWORDS = {"拍照", "拍摄", "拍一张"};

    private final Context context;
    private final Listener listener;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    private AudioManager audioManager;
    private TextToSpeech textToSpeech;
    private boolean ttsReady = false;
    private boolean scoStarted = false;
    private ToneGenerator toneGenerator;

    private AudioRecord audioRecord;
    private volatile boolean recording = false;
    private volatile boolean pendingStart = false;
    private ByteArrayOutputStream pcmBuffer;
    private Thread recordThread;

    // 讯飞 SparkChain 语音听写
    private ASR mAsr;
    private volatile boolean asrInProgress = false;
    private volatile boolean asrFinished = false;

    public VoiceController(Context context, Listener listener) {
        this.context = context.getApplicationContext();
        this.listener = listener;
        this.audioManager = (AudioManager) this.context.getSystemService(Context.AUDIO_SERVICE);
        initTts();
        toneGenerator = new ToneGenerator(AudioManager.STREAM_MUSIC, 80);
    }

    public boolean isRecording() {
        return recording;
    }

    /** 按住说话：先播开始提示音，再启动蓝牙 SCO 采集眼镜麦克风 */
    public void startCapture() {
        if (recording || pendingStart) return;
        pendingStart = true;
        // 在进入通话模式（SCO）前播放提示音，此时媒体流正常路由：连眼镜走眼镜，否则走手机扬声器
        playStartTone();
        mainHandler.postDelayed(() -> {
            pendingStart = false;
            beginRecording();
        }, 200);
    }

    private void beginRecording() {
        if (recording) return;
        startSco();

        int minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        int bufSize = Math.max(minBuf * 2, 6400);
        try {
            audioRecord = new AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION,
                    SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufSize);
        } catch (Exception e) {
            Log.e(TAG, "create AudioRecord error", e);
            notifyError("无法创建录音器，请确认已授予麦克风权限");
            stopSco();
            return;
        }

        pcmBuffer = new ByteArrayOutputStream();
        recording = true;
        try {
            audioRecord.startRecording();
        } catch (Exception e) {
            Log.e(TAG, "startRecording error", e);
            recording = false;
            audioRecord.release();
            audioRecord = null;
            pcmBuffer = null;
            notifyError("录音启动失败：" + e.getMessage());
            stopSco();
            return;
        }

        recordThread = new Thread(() -> {
            byte[] buf = new byte[3200];
            while (recording) {
                int n = audioRecord.read(buf, 0, buf.length);
                if (n > 0) {
                    synchronized (pcmBuffer) {
                        pcmBuffer.write(buf, 0, n);
                    }
                }
            }
        });
        recordThread.start();
        notifyListening(true);
    }

    /** 松开：停止录音，交给讯飞 SparkChain 识别 */
    public void stopCaptureAndRecognize() {
        if (pendingStart) {
            pendingStart = false;
            return; // 提示音还没播完就松手了，取消本次录音
        }
        if (!recording) return;
        recording = false;

        if (recordThread != null) {
            try {
                recordThread.join(300);
            } catch (InterruptedException ignored) {}
            recordThread = null;
        }

        byte[] pcm;
        synchronized (pcmBuffer) {
            pcm = pcmBuffer.toByteArray();
        }

        try {
            audioRecord.stop();
        } catch (Exception ignored) {}
        try {
            audioRecord.release();
        } catch (Exception ignored) {}
        audioRecord = null;
        pcmBuffer = null;
        // 立即退出通话模式（SCO），在媒体流下播放结束提示音
        stopSco();
        playStopTone();
        notifyListening(false);

        if (pcm.length < 3200) { // 少于 0.1 秒，判为没录到
            notifyError("没录到声音，请按住后说话");
            return;
        }
        recognizeWithSparkChain(pcm);
    }

    /** 播报文本到眼镜扬声器 */
    public void speak(final String text) {
        if (text == null || text.isEmpty()) return;
        if (!ttsReady || textToSpeech == null) {
            mainHandler.postDelayed(() -> speakNow(text), 600);
            return;
        }
        speakNow(text);
    }

    private void speakNow(String text) {
        if (textToSpeech == null || !ttsReady) {
            Log.w(TAG, "TTS 未就绪，跳过播报");
            return;
        }
        // 走媒体音频流（A2DP），音频自动路由到蓝牙眼镜扬声器，不要用通话通道 SCO
        Bundle params = new Bundle();
        params.putInt(TextToSpeech.Engine.KEY_PARAM_STREAM, AudioManager.STREAM_MUSIC);
        textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, params, "ar_glass_tts");
    }

    /** 释放资源（Activity onDestroy 调用） */
    public void release() {
        if (recording) {
            recording = false;
            try {
                audioRecord.stop();
            } catch (Exception ignored) {}
            try {
                audioRecord.release();
            } catch (Exception ignored) {}
            audioRecord = null;
        }
        if (mAsr != null && asrInProgress) {
            try {
                mAsr.stop(true);
            } catch (Exception ignored) {}
        }
        if (textToSpeech != null) {
            try {
                textToSpeech.stop();
                textToSpeech.shutdown();
            } catch (Exception ignored) {}
            textToSpeech = null;
        }
        ttsReady = false;
        if (toneGenerator != null) {
            try {
                toneGenerator.release();
            } catch (Exception ignored) {}
            toneGenerator = null;
        }
        stopSco();
    }

    // ========== 讯飞 SparkChain 语音听写 ==========

    private void recognizeWithSparkChain(byte[] pcm) {
        if (asrInProgress) {
            notifyError("上一次识别尚未完成");
            return;
        }
        asrInProgress = true;
        asrFinished = false;

        if (mAsr == null) {
            mAsr = new ASR();
            mAsr.registerCallbacks(new AsrCallbacks() {
                @Override
                public void onResult(ASR.ASRResult asrResult, Object o) {
                    // status：0 第一块结果，1 中间结果，2 最后一块结果
                    int status = asrResult.getStatus();
                    String text = asrResult.getBestMatchText();
                    if (status == 2) {
                        handleAsrEnd(text, null);
                    }
                }

                @Override
                public void onError(ASR.ASRError asrError, Object o) {
                    String msg = asrError.getErrMsg();
                    int code = asrError.getCode();
                    handleAsrEnd(null, "识别失败(code=" + code + ")：" + msg);
                }

                @Override
                public void onBeginOfSpeech() {}

                @Override
                public void onEndOfSpeech() {}
            });
        }

        mAsr.language("zh_cn");
        mAsr.domain("iat");
        mAsr.accent("mandarin");
        mAsr.vinfo(true);

        int ret = mAsr.start("voice_" + System.currentTimeMillis());
        if (ret != 0) {
            handleAsrEnd(null, "识别启动失败，错误码：" + ret);
            return;
        }

        // 后台线程分帧写入音频，写完后 stop 通知云端
        new Thread(() -> {
            final int FRAME = 6400; // 约 200ms
            int offset = 0;
            while (offset < pcm.length && !asrFinished) {
                int len = Math.min(FRAME, pcm.length - offset);
                byte[] slice = Arrays.copyOfRange(pcm, offset, offset + len);
                int wr = mAsr.write(slice);
                if (wr != 0) {
                    Log.w(TAG, "ASR write ret=" + wr);
                    break;
                }
                offset += len;
                try {
                    Thread.sleep(40);
                } catch (InterruptedException e) {
                    break;
                }
            }
            mAsr.stop(false); // 音频输入完毕，等云端最后一包下发
        }).start();

        // 超时兜底：8 秒内没拿到最终结果就主动结束
        mainHandler.postDelayed(() -> {
            if (!asrFinished && asrInProgress) {
                handleAsrEnd(null, "识别超时，请重试");
            }
        }, 8000);
    }

    private void handleAsrEnd(String text, String errorMsg) {
        if (asrFinished) return;
        asrFinished = true;
        asrInProgress = false;

        final String t = (text == null) ? "" : text.trim();
        final String err = errorMsg;
        mainHandler.post(() -> {
            if (err != null && !err.isEmpty()) {
                notifyError(err);
                return;
            }
            if (t.isEmpty()) {
                notifyError("未识别到内容，请重试");
                return;
            }
            listener.onSpeechText(t);

            boolean hit = false;
            for (String kw : KEYWORDS) {
                if (t.contains(kw)) {
                    hit = true;
                    break;
                }
            }
            if (hit) {
                listener.onKeywordDetected(t);
            }
        });
    }

    // ========== TTS 与 SCO ==========

    private void initTts() {
        textToSpeech = new TextToSpeech(context, status -> {
            if (status == TextToSpeech.SUCCESS) {
                int r = textToSpeech.setLanguage(Locale.CHINA);
                ttsReady = (r != TextToSpeech.LANG_MISSING_DATA && r != TextToSpeech.LANG_NOT_SUPPORTED);
            }
        });
    }

    private void startSco() {
        if (audioManager == null || scoStarted) return;
        try {
            audioManager.setMode(AudioManager.MODE_IN_COMMUNICATION);
            audioManager.setBluetoothScoOn(true);
            audioManager.startBluetoothSco();
            scoStarted = true;
        } catch (Exception e) {
            Log.e(TAG, "startSco error", e);
        }
    }

    private void stopSco() {
        if (audioManager == null || !scoStarted) return;
        try {
            audioManager.setBluetoothScoOn(false);
            audioManager.stopBluetoothSco();
            audioManager.setMode(AudioManager.MODE_NORMAL);
        } catch (Exception e) {
            Log.e(TAG, "stopSco error", e);
        }
        scoStarted = false;
    }

    private void notifyListening(final boolean value) {
        mainHandler.post(() -> listener.onListeningChanged(value));
    }

    private void notifyError(final String message) {
        mainHandler.post(() -> listener.onError(message));
    }

    /** 录音开始提示音（单声“滴”） */
    private void playStartTone() {
        try {
            if (toneGenerator != null) {
                toneGenerator.startTone(ToneGenerator.TONE_PROP_BEEP, 150);
            }
        } catch (Exception e) {
            Log.e(TAG, "playStartTone error", e);
        }
    }

    /** 录音结束提示音（双声“滴滴”） */
    private void playStopTone() {
        try {
            if (toneGenerator != null) {
                toneGenerator.startTone(ToneGenerator.TONE_PROP_BEEP2, 200);
            }
        } catch (Exception e) {
            Log.e(TAG, "playStopTone error", e);
        }
    }
}
