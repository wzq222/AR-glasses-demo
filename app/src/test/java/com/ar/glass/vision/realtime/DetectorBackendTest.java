package com.ar.glass.vision.realtime;

import android.graphics.Bitmap;

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
        assertEquals(640, NcnnFastenerDetector.INPUT_SIZE);
        assertEquals(6, NcnnFastenerDetector.OUTPUT_CHANNELS);
        assertEquals(34_000, NcnnFastenerDetector.OUTPUT_CANDIDATES);
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
