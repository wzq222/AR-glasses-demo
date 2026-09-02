package com.ar.glass.vision.realtime;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;

import org.junit.Test;

public class OnnxWitnessStateEstimatorTest {
    @Test public void decodesAlignedAndHighSuspicionGeometry() {
        WitnessStateEstimate aligned = decode(new int[][]{
                {80, 160}, {145, 160}, {145, 160}, {220, 160}});
        assertEquals(WitnessTriage.LIKELY_ALIGNED, aligned.getTriage());
        assertEquals(0f, aligned.getAngleDegrees(), 0.01f);

        WitnessStateEstimate displaced = decode(new int[][]{
                {80, 160}, {145, 160}, {145, 160}, {215, 190}});
        assertEquals(WitnessTriage.HIGH_SUSPICION, displaced.getTriage());
        assertTrue(displaced.getAngleDegrees() > 15f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsUnlocalizedHeatmaps() {
        float[][][][] segmentation = tensor(-1f);
        float[][][][] keypoints = tensor(0f);
        for (int y = 150; y < 170; y++) segmentation[0][2][y][150] = 1f;
        OnnxWitnessStateEstimator.decodeOutputs(
                segmentation, keypoints, new float[][]{{0f, 0f, 0f, 0f}}, 1.0);
    }

    private static WitnessStateEstimate decode(int[][] points) {
        float[][][][] segmentation = tensor(-1f);
        float[][][][] keypoints = tensor(-20f);
        for (int channel = 0; channel < 4; channel++) {
            int x = points[channel][0];
            int y = points[channel][1];
            keypoints[0][channel][y][x] = 20f;
            for (int yy = Math.max(0, y - 2); yy <= Math.min(319, y + 2); yy++) {
                for (int xx = Math.max(0, x - 2); xx <= Math.min(319, x + 2); xx++) {
                    segmentation[0][2][yy][xx] = 1f;
                }
            }
        }
        return OnnxWitnessStateEstimator.decodeOutputs(
                segmentation, keypoints, new float[][]{{0f, 0f, 0f, 0f}}, 2.5);
    }

    private static float[][][][] tensor(float value) {
        float[][][][] tensor = new float[1][4][320][320];
        for (float[][] channel : tensor[0]) {
            for (float[] row : channel) Arrays.fill(row, value);
        }
        return tensor;
    }
}
