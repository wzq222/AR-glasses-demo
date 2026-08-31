package com.ar.glass.vision.realtime;

public final class InferenceGate {
    private final long minimumIntervalMillis;
    private boolean inFlight;
    private long lastStartedAtMillis = Long.MIN_VALUE;

    public InferenceGate(long minimumIntervalMillis) {
        if (minimumIntervalMillis < 0L) {
            throw new IllegalArgumentException("minimum interval must not be negative");
        }
        this.minimumIntervalMillis = minimumIntervalMillis;
    }

    public synchronized boolean tryAcquire(long nowMillis) {
        if (inFlight) {
            return false;
        }
        if (lastStartedAtMillis != Long.MIN_VALUE
                && nowMillis - lastStartedAtMillis < minimumIntervalMillis) {
            return false;
        }
        inFlight = true;
        lastStartedAtMillis = nowMillis;
        return true;
    }

    public synchronized void release() {
        inFlight = false;
    }
}
