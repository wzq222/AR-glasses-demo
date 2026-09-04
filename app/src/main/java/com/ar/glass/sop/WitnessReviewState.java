package com.ar.glass.sop;

/** Keeps one operator decision per detected witness-line point. Zero means unreviewed. */
final class WitnessReviewState {
    private final int[] decisions;
    private int currentIndex;

    WitnessReviewState(int pointCount) {
        if (pointCount <= 0) {
            throw new IllegalArgumentException("pointCount must be positive");
        }
        decisions = new int[pointCount];
    }

    int getCurrentIndex() {
        return currentIndex;
    }

    int getCurrentDecision() {
        return decisions[currentIndex];
    }

    int getDecision(int index) {
        return decisions[index];
    }

    void setCurrentDecision(int decision) {
        if (decision < 0 || decision > 3) {
            throw new IllegalArgumentException("decision must be between 0 and 3");
        }
        decisions[currentIndex] = decision;
    }

    boolean movePrevious() {
        if (currentIndex == 0) return false;
        currentIndex--;
        return true;
    }

    boolean moveNext() {
        if (currentIndex >= decisions.length - 1) return false;
        currentIndex++;
        return true;
    }

    boolean allReviewed() {
        for (int decision : decisions) {
            if (decision == 0) return false;
        }
        return true;
    }

    int size() {
        return decisions.length;
    }
}
