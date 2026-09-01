package com.ar.glass.util;

import android.content.Context;
import android.speech.tts.TextToSpeech;
import android.util.Log;

import java.util.Locale;

/**
 * 语音播报封装（TextToSpeech）。
 *
 * 用法：构造后调用 {@link #speak(String)}；播报文本建议先用
 * {@link com.ar.glass.vision.MeterReading#getSpeechText()} 转成中文单位。
 */
public class MeterTts implements TextToSpeech.OnInitListener {

    private static final String TAG = "MeterTts";

    private final TextToSpeech tts;
    private volatile boolean ready = false;

    public MeterTts(Context context) {
        tts = new TextToSpeech(context.getApplicationContext(), this);
    }

    @Override
    public void onInit(int status) {
        if (status == TextToSpeech.SUCCESS) {
            int res = tts.setLanguage(Locale.CHINESE);
            if (res == TextToSpeech.LANG_MISSING_DATA || res == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.w(TAG, "中文语音不可用，回退到系统默认语言");
                tts.setLanguage(Locale.getDefault());
            }
            ready = true;
        } else {
            Log.e(TAG, "TTS 初始化失败：" + status);
        }
    }

    public boolean isReady() {
        return ready;
    }

    /** 立即播报（QUEUE_FLUSH：打断上一条，保证读到最新结果）。 */
    public void speak(String text) {
        if (!ready || text == null || text.isEmpty()) {
            return;
        }
        tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "meter");
    }

    public void shutdown() {
        try {
            tts.stop();
            tts.shutdown();
        } catch (Exception ignored) {
        }
    }
}
