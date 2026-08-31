package com.ar.glass.vision.realtime;

public final class InferenceGate {
    private final long minimumIntervalMillis;
    private boolean inFlight;
    private long lastCompletedAtMillis = Long.MIN_VALUE;

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
        if (lastCompletedAtMillis != Long.MIN_VALUE
                && nowMillis - lastCompletedAtMillis < minimumIntervalMillis) {
            return false;
        }
        inFlight = true;
        return true;
    }

    public synchronized void release(long completedAtMillis) {
        inFlight = false;
        lastCompletedAtMillis = completedAtMillis;
    }
}
