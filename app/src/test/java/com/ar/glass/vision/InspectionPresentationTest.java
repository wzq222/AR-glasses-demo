package com.ar.glass.vision;

import com.ar.glass.vision.realtime.WitnessStateEstimate;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class InspectionPresentationTest {
    private static final float[] POINTS = {
            0.20f, 0.50f, 0.45f, 0.50f,
            0.46f, 0.50f, 0.75f, 0.50f
    };

    @Test
    public void identifiesTheRelevantBoltBeforeReportingItsState() {
        assertEquals("螺栓 3（有防松标记）", InspectionPresentation.boltLabel(3));
        assertEquals(
                "未见松动迹象 · 夹角 2.5°",
                InspectionPresentation.stateLabel(
                        WitnessStateEstimate.measured(2.5f, POINTS, 1.0)));
        assertEquals(
                "疑似松动 · 夹角 8.0° · 请审核",
                InspectionPresentation.stateLabel(
                        WitnessStateEstimate.measured(8f, POINTS, 1.0)));
        assertEquals(
                "高疑似松动 · 夹角 18.0° · 请审核",
                InspectionPresentation.stateLabel(
                        WitnessStateEstimate.measured(18f, POINTS, 1.0)));
    }

    @Test
    public void refusesToInventALoosenessStateWithoutGeometry() {
        assertEquals(
                "无法判断 · 请近拍审核",
                InspectionPresentation.stateLabel(
                        WitnessStateEstimate.insufficient("LOW_RESOLUTION")));
    }
}
