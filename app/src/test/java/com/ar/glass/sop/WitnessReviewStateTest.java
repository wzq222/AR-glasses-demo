package com.ar.glass.sop;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class WitnessReviewStateTest {
    @Test
    public void preservesEachDecisionAcrossNavigation() {
        WitnessReviewState state = new WitnessReviewState(3);

        state.setCurrentDecision(1);
        assertTrue(state.moveNext());
        state.setCurrentDecision(3);
        assertTrue(state.movePrevious());

        assertEquals(0, state.getCurrentIndex());
        assertEquals(1, state.getCurrentDecision());
        assertEquals(3, state.getDecision(1));
        assertFalse(state.allReviewed());
    }

    @Test
    public void reportsCompletionOnlyAfterEveryPointIsReviewed() {
        WitnessReviewState state = new WitnessReviewState(2);

        state.setCurrentDecision(1);
        state.moveNext();
        assertFalse(state.allReviewed());
        state.setCurrentDecision(2);

        assertTrue(state.allReviewed());
    }

    @Test
    public void navigationStopsAtBothEnds() {
        WitnessReviewState state = new WitnessReviewState(1);

        assertFalse(state.movePrevious());
        assertFalse(state.moveNext());
        assertEquals(0, state.getCurrentIndex());
    }
}
