package com.ar.glass.vision.realtime;

final class InferenceThreadPolicy {
    private InferenceThreadPolicy() {
    }

    static int intraOpThreads() {
        return 4;
    }
}
