package com.ar.glass.vision.ui;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class WitnessReviewCropTest {
    @Test
    public void usesTwoPointTwoFiveExpansionAndCentersRoi() {
        WitnessReviewCrop crop = WitnessReviewCrop.fromNormalized(
                0.40f, 0.30f, 0.45f, 0.36666667f, 2000, 1500);

        assertEquals(225f, crop.getSide(), 0.01f);
        assertEquals(0.5f, (crop.getRoiLeft() + crop.getRoiRight()) / 2f, 0.001f);
        assertEquals(0.5f, (crop.getRoiTop() + crop.getRoiBottom()) / 2f, 0.001f);
        assertFalse(crop.requiresCloserCapture());
    }

    @Test
    public void keepsTargetCenteredAtImageEdgeUsingPadding() {
        WitnessReviewCrop crop = WitnessReviewCrop.fromNormalized(
                0f, 0f, 0.04f, 0.04f, 2000, 1500);

        assertTrue(crop.getRequestedLeft() < 0f);
        assertTrue(crop.getRequestedTop() < 0f);
        assertEquals(0.5f, (crop.getRoiLeft() + crop.getRoiRight()) / 2f, 0.001f);
        assertEquals(0.5f, (crop.getRoiTop() + crop.getRoiBottom()) / 2f, 0.001f);
    }

    @Test
    public void flagsTargetsSmallerThanThirtyTwoSourcePixels() {
        WitnessReviewCrop crop = WitnessReviewCrop.fromNormalized(
                0.20f, 0.20f, 0.21f, 0.21f, 2000, 1500);

        assertTrue(crop.requiresCloserCapture());
    }

    @Test
    public void rejectsInvalidBounds() {
        assertThrows(IllegalArgumentException.class, () ->
                WitnessReviewCrop.fromNormalized(0.5f, 0.2f, 0.4f, 0.3f, 2000, 1500));
    }
}
