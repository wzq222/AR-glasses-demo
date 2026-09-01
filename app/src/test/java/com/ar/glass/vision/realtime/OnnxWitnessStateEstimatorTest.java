package com.ar.glass.vision.realtime;

import com.ar.glass.vision.fastener.WitnessReviewHint;
import com.ar.glass.vision.fastener.WitnessStateEstimate;

import org.junit.Test;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

public class OnnxWitnessStateEstimatorTest {
    @Test
    public void pinsTheExperimentalModelContract() {
        assertEquals("witness-roi.onnx", OnnxWitnessStateEstimator.MODEL_ASSET_NAME);
        assertEquals(320, OnnxWitnessStateEstimator.INPUT_SIZE);

        Map<String, long[]> outputs = new LinkedHashMap<>();
        outputs.put("segmentation_logits", new long[]{-1, 4, 320, 320});
        outputs.put("keypoint_heatmaps", new long[]{-1, 4, 320, 320});
        outputs.put("quality_logits", new long[]{-1, 4});
        OnnxWitnessStateEstimator.validateModelShapes(
                new long[]{-1, 3, 320, 320}, outputs);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsUnexpectedOutputShape() {
        Map<String, long[]> outputs = new LinkedHashMap<>();
        outputs.put("segmentation_logits", new long[]{-1, 4, 320, 320});
        outputs.put("keypoint_heatmaps", new long[]{-1, 4, 160, 160});
        outputs.put("quality_logits", new long[]{-1, 4});
        OnnxWitnessStateEstimator.validateModelShapes(
                new long[]{-1, 3, 320, 320}, outputs);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsOutputsWithDifferentBatchContract() {
        Map<String, long[]> outputs = new LinkedHashMap<>();
        outputs.put("segmentation_logits", new long[]{1, 4, 320, 320});
        outputs.put("keypoint_heatmaps", new long[]{-1, 4, 320, 320});
        outputs.put("quality_logits", new long[]{-1, 4});
        OnnxWitnessStateEstimator.validateModelShapes(
                new long[]{-1, 3, 320, 320}, outputs);
    }

    @Test
    public void decodesHeatmapChannelsAndAbsoluteSegmentAngle() {
        float[][][] heatmaps = new float[4][320][320];
        heatmaps[0][10][10] = 1f;  // fixed_outer
        heatmaps[1][10][110] = 2f; // fixed_joint
        heatmaps[2][20][110] = 3f; // moving_joint
        heatmaps[3][120][110] = 4f;// moving_outer

        WitnessStateEstimate result = OnnxWitnessStateEstimator.decodeEstimate(
                heatmaps, 7.25);

        assertEquals(90f, result.getAngle().getPointEstimateDegrees(), 0.0001f);
        assertEquals(WitnessReviewHint.POSSIBLE_DISPLACED, result.getReviewHint());
        assertEquals("POINT_ANGLE_SECOND_VIEW_REQUIRED", result.getReviewReason());
        assertEquals(7.25, result.getInferenceMillis(), 0f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void degenerateDecodedSegmentFailsClosed() {
        float[][][] heatmaps = new float[4][320][320];
        heatmaps[0][10][10] = 1f;
        heatmaps[1][10][10] = 2f;
        heatmaps[2][20][20] = 3f;
        heatmaps[3][30][30] = 4f;

        OnnxWitnessStateEstimator.decodeEstimate(heatmaps, 1.0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void nonfiniteHeatmapFailsClosed() {
        float[][][] heatmaps = new float[4][320][320];
        heatmaps[0][10][10] = Float.NaN;

        OnnxWitnessStateEstimator.decodeEstimate(heatmaps, 1.0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void nonfiniteUnusedQualityOutputAlsoFailsClosed() {
        float[][][][] segmentation = new float[1][4][320][320];
        float[][][][] keypoints = new float[1][4][320][320];
        float[][] quality = new float[][]{{0f, 0f, Float.NaN, 0f}};

        OnnxWitnessStateEstimator.validateDecodedOutputs(
                segmentation, keypoints, quality);
    }

    @Test
    public void acceptsMinimumSemanticallyUsableEvidence() {
        Evidence evidence = semanticallyUsableEvidence();

        OnnxWitnessStateEstimator.validateDecodedOutputs(
                evidence.segmentation, evidence.keypoints, evidence.quality);
    }

    @Test(expected = IllegalArgumentException.class)
    public void emptyWitnessMaskFailsClosed() {
        Evidence evidence = semanticallyUsableEvidence();
        for (int y = 0; y < 320; y++) {
            java.util.Arrays.fill(evidence.segmentation[0][2][y], -1f);
        }

        OnnxWitnessStateEstimator.validateDecodedOutputs(
                evidence.segmentation, evidence.keypoints, evidence.quality);
    }

    @Test(expected = IllegalArgumentException.class)
    public void lowMarkIntegrityQualityFailsClosed() {
        Evidence evidence = semanticallyUsableEvidence();
        evidence.quality[0][0] = -0.01f;

        OnnxWitnessStateEstimator.validateDecodedOutputs(
                evidence.segmentation, evidence.keypoints, evidence.quality);
    }

    @Test(expected = IllegalArgumentException.class)
    public void highOcclusionQualityFailsClosed() {
        Evidence evidence = semanticallyUsableEvidence();
        evidence.quality[0][1] = 0.01f;

        OnnxWitnessStateEstimator.validateDecodedOutputs(
                evidence.segmentation, evidence.keypoints, evidence.quality);
    }

    @Test(expected = IllegalArgumentException.class)
    public void flatKeypointHeatmapFailsClosed() {
        Evidence evidence = semanticallyUsableEvidence();
        evidence.keypoints[0][3][120][110] = 0f;

        OnnxWitnessStateEstimator.validateDecodedOutputs(
                evidence.segmentation, evidence.keypoints, evidence.quality);
    }

    private static Evidence semanticallyUsableEvidence() {
        float[][][][] segmentation = new float[1][4][320][320];
        for (int channel = 0; channel < 4; channel++) {
            for (int y = 0; y < 320; y++) {
                java.util.Arrays.fill(segmentation[0][channel][y], -1f);
            }
        }
        for (int index = 0; index < 8; index++) {
            segmentation[0][2][10][index] = 0f;
        }
        float[][][][] keypoints = new float[1][4][320][320];
        keypoints[0][0][10][10] = 1f;
        keypoints[0][1][10][110] = 1f;
        keypoints[0][2][20][110] = 1f;
        keypoints[0][3][120][110] = 1f;
        float[][] quality = new float[][]{{0f, 0f, 0f, 0f}};
        return new Evidence(segmentation, keypoints, quality);
    }

    private static final class Evidence {
        private final float[][][][] segmentation;
        private final float[][][][] keypoints;
        private final float[][] quality;

        private Evidence(
                float[][][][] segmentation,
                float[][][][] keypoints,
                float[][] quality) {
            this.segmentation = segmentation;
            this.keypoints = keypoints;
            this.quality = quality;
        }
    }
}
