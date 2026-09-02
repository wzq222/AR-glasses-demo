package com.ar.glass.sop;

public final class SopStepPolicy {
    private SopStepPolicy() {}

    public static String statusForDecision(int decisionIndex) {
        return decisionIndex == 0 ? "succeeded" : "uncertain";
    }

    public static String humanDecision(int decisionIndex) {
        if (decisionIndex == 0) return "confirmed_ok";
        if (decisionIndex == 1) return "suspected";
        return "unable_to_judge";
    }

    public static String idempotencyKey(String runId, String stepKey) {
        return runId + ":" + stepKey;
    }
}
