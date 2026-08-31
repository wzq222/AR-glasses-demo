package com.ar.glass.vision.fastener;

public final class ProvisionalTriageThresholds {
    private final float reviewDegrees;
    private final float highSuspicionDegrees;

    public ProvisionalTriageThresholds(float reviewDegrees, float highSuspicionDegrees) {
        if (!isFinite(reviewDegrees)
                || !isFinite(highSuspicionDegrees)
                || reviewDegrees <= 0f
                || reviewDegrees >= highSuspicionDegrees
                || highSuspicionDegrees > 90f) {
            throw new IllegalArgumentException(
                    "provisional triage thresholds are outside valid ranges");
        }
        this.reviewDegrees = reviewDegrees;
        this.highSuspicionDegrees = highSuspicionDegrees;
    }

    public static ProvisionalTriageThresholds defaults() {
        return new ProvisionalTriageThresholds(3f, 15f);
    }

    public float getReviewDegrees() { return reviewDegrees; }
    public float getHighSuspicionDegrees() { return highSuspicionDegrees; }

    private static boolean isFinite(float value) {
        return !Float.isNaN(value) && !Float.isInfinite(value);
    }
}
