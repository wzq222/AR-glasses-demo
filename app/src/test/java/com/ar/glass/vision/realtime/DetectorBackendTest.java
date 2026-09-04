package com.ar.glass.vision.realtime;

import android.graphics.Bitmap;
import android.content.Context;

import org.junit.Test;

import java.io.Closeable;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.FloatBuffer;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class DetectorBackendTest {
    @Test
    public void parsesOnlyTheTwoSupportedBackends() {
        assertEquals(DetectorBackend.ONNX, DetectorBackend.fromBuildValue(null));
        assertEquals(DetectorBackend.ONNX, DetectorBackend.fromBuildValue(""));
        assertEquals(DetectorBackend.ONNX, DetectorBackend.fromBuildValue("onnx"));
        assertEquals(DetectorBackend.NCNN, DetectorBackend.fromBuildValue("NCNN"));
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsAnUnknownBackendInsteadOfSilentlyChangingRuntime() {
        DetectorBackend.fromBuildValue("mnn");
    }

    @Test
    public void bothImplementationsExposeOneSharedLifecycle() throws Exception {
        assertTrue(Closeable.class.isAssignableFrom(FastenerDetector.class));
        assertTrue(FastenerDetector.class.isAssignableFrom(OnnxFastenerDetector.class));
        assertTrue(FastenerDetector.class.isAssignableFrom(NcnnFastenerDetector.class));
        assertEquals(OnnxFastenerDetector.DetectionResult.class,
                FastenerDetector.class.getMethod("detect", Bitmap.class).getReturnType());
    }

    @Test
    public void ncnnPinsTheParityValidatedTensorContract() {
        assertEquals("model.ncnn.param", NcnnFastenerDetector.PARAM_ASSET_NAME);
        assertEquals("model.ncnn.bin", NcnnFastenerDetector.BIN_ASSET_NAME);
        assertEquals("in0", NcnnFastenerDetector.INPUT_BLOB_NAME);
        assertEquals("out0", NcnnFastenerDetector.OUTPUT_BLOB_NAME);
        assertEquals(512, NcnnFastenerDetector.INPUT_SIZE);
        assertEquals(5, NcnnFastenerDetector.OUTPUT_CHANNELS);
        assertEquals(21_760, NcnnFastenerDetector.OUTPUT_CANDIDATES);
        assertEquals(0.70f, NcnnFastenerDetector.NMS_IOU_THRESHOLD, 0f);
        assertEquals(300, NcnnFastenerDetector.MAX_DETECTIONS);
    }

    @Test
    public void ncnnExposesExplicitVulkanAndFp16Selections() throws Exception {
        assertEquals(NcnnFastenerDetector.class,
                NcnnFastenerDetector.class
                        .getConstructor(
                                Context.class,
                                float.class,
                                boolean.class,
                                boolean.class)
                        .getDeclaringClass());
    }

    @Test
    public void bothBackendsUseTheSameReusablePreprocessor() throws Exception {
        assertWorkspaceField(OnnxFastenerDetector.class);
        assertWorkspaceField(NcnnFastenerDetector.class);
    }

    @Test
    public void flatNativeOutputUsesTheSamePostprocessorPolicy() {
        FloatBuffer prediction = FloatBuffer.allocate(6 * 2);
        putRow(prediction, 0, 2, 100f, 300f);
        putRow(prediction, 1, 2, 100f, 300f);
        putRow(prediction, 2, 2, 40f, 40f);
        putRow(prediction, 3, 2, 20f, 20f);
        putRow(prediction, 4, 2, 0.30f, 0.70f);
        putRow(prediction, 5, 2, 0.80f, 0.20f);

        List<Detection> detections = YoloPostprocessor.process(
                prediction, 2, 640, 640, 1f, 0f, 0f);

        assertEquals(2, detections.size());
        assertEquals(1, detections.get(0).getClassId());
        assertEquals(0, detections.get(1).getClassId());
    }

    @Test
    public void flatNativeOutputSupportsAnExplicitRecallCalibrationThreshold() {
        FloatBuffer prediction = FloatBuffer.allocate(6);
        prediction.put(0, 100f);
        prediction.put(1, 100f);
        prediction.put(2, 20f);
        prediction.put(3, 20f);
        prediction.put(4, 0.195f);
        prediction.put(5, 0.10f);

        assertEquals(0, YoloPostprocessor.process(
                prediction, 1, 640, 640, 1f, 0f, 0f).size());
        assertEquals(1, YoloPostprocessor.process(
                prediction, 1, 640, 640, 1f, 0f, 0f, 0.19f, 0.45f).size());
    }

    @Test
    public void flatNativeOutputSupportsTheSingleClassMarkedPointContract() {
        FloatBuffer prediction = FloatBuffer.allocate(5);
        prediction.put(0, 100f);
        prediction.put(1, 100f);
        prediction.put(2, 20f);
        prediction.put(3, 20f);
        prediction.put(4, 0.75f);

        List<Detection> detections = YoloPostprocessor.process(
                prediction,
                5,
                1,
                512,
                512,
                1f,
                0f,
                0f,
                0.001f,
                0.70f,
                1_000,
                300);

        assertEquals(1, detections.size());
        assertEquals(0, detections.get(0).getClassId());
        assertEquals(0.75f, detections.get(0).getConfidence(), 0f);
    }

    @Test
    public void markedPointVerifierPinsTheValidatedBatchContract() {
        assertEquals("marked-point-verifier.onnx", MarkedPointOnnxVerifier.MODEL_ASSET_NAME);
        assertEquals(128, MarkedPointOnnxVerifier.INPUT_SIZE);
        assertEquals(0.28198338f, MarkedPointOnnxVerifier.VERIFIER_THRESHOLD, 1.0e-7f);
        assertEquals(0.96769285f, MarkedPointOnnxVerifier.PROPOSAL_BYPASS_THRESHOLD, 1.0e-7f);
        assertEquals(0.30f, MarkedPointOnnxVerifier.NMS_IOU_THRESHOLD, 0f);
    }

    @Test
    public void markedPointVerifierExposesAnExplicitNnapiExperiment() throws Exception {
        assertEquals(
                MarkedPointOnnxVerifier.class,
                MarkedPointOnnxVerifier.class
                        .getConstructor(Context.class, boolean.class)
                        .getDeclaringClass());
        assertEquals(
                MarkedPointOnnxVerifier.class,
                MarkedPointOnnxVerifier.class
                        .getConstructor(Context.class, boolean.class, boolean.class)
                        .getDeclaringClass());
    }

    @Test
    public void markedPointVerifierKeepsVerifierOrProposalEvidenceAndSuppressesDuplicates() {
        List<Detection> proposals = java.util.Arrays.asList(
                new Detection(10f, 10f, 50f, 50f, 0.20f, 0),
                new Detection(12f, 12f, 52f, 52f, 0.10f, 0),
                new Detection(100f, 100f, 140f, 140f, 0.98f, 0),
                new Detection(200f, 200f, 240f, 240f, 0.10f, 0));

        List<Detection> selected = MarkedPointOnnxVerifier.filterDetections(
                proposals, new float[]{0.90f, 0.80f, 0.10f, 0.20f});

        assertEquals(2, selected.size());
        assertEquals(10f, selected.get(0).getLeft(), 0f);
        assertEquals(100f, selected.get(1).getLeft(), 0f);
    }

    @Test
    public void markedPointVerifierHidesBroadContextBoxWhenTightCandidateExists() {
        List<Detection> proposals = java.util.Arrays.asList(
                new Detection(0f, 0f, 240f, 240f, 0.99f, 0),
                new Detection(80f, 80f, 140f, 140f, 0.20f, 0));

        List<Detection> selected = MarkedPointOnnxVerifier.filterDetections(
                proposals, new float[]{0.10f, 0.90f});

        assertEquals(1, selected.size());
        assertEquals(80f, selected.get(0).getLeft(), 0f);
        assertEquals(140f, selected.get(0).getRight(), 0f);
    }

    @Test
    public void markedPointVerifierCropMatchesTheTrainingContextRule() {
        assertTrue(java.util.Arrays.equals(
                new int[]{68, 28, 64},
                MarkedPointOnnxVerifier.computeSquareCrop(
                        new Detection(80f, 40f, 120f, 80f, 0.5f, 0), 300, 200)));
    }

    @Test
    public void factoryWrapsNcnnWithTheMarkedPointVerifierOnlyWhenEnabled() {
        FastenerDetector plain = DetectorFactory.create(
                null, "ncnn", 0.0019424824f, false, false, false);
        FastenerDetector verified = DetectorFactory.create(
                null, "ncnn", 0.0019424824f, false, false, true);

        assertEquals(NcnnFastenerDetector.class, plain.getClass());
        assertEquals(VerifiedNcnnFastenerDetector.class, verified.getClass());
        plain.close();
        verified.close();
    }

    private static void putRow(
            FloatBuffer buffer, int row, int columns, float first, float second) {
        buffer.put(row * columns, first);
        buffer.put(row * columns + 1, second);
    }

    private static void assertWorkspaceField(Class<?> detectorClass) throws Exception {
        Field field = detectorClass.getDeclaredField("inputWorkspace");
        assertEquals(FastenerInputWorkspace.class, field.getType());
        assertTrue(Modifier.isFinal(field.getModifiers()));
    }
}
