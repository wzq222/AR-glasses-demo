package com.ar.glass.vision.realtime;

import android.graphics.Bitmap;

import org.junit.Test;

import java.io.Closeable;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class ModelAssetContractTest {
    private static final float EPSILON = 0.0001f;

    @Test
    public void exposesTheFrozenModelContract() {
        assertEquals("fastener-target-p2-640.onnx", OnnxFastenerDetector.MODEL_ASSET_NAME);
        assertEquals(640, OnnxFastenerDetector.INPUT_SIZE);
        assertEquals(6, OnnxFastenerDetector.OUTPUT_CHANNELS);
        assertEquals(34_000, OnnxFastenerDetector.OUTPUT_CANDIDATES);
    }

    @Test
    public void exposesTheDetectorLifecycleApi() throws Exception {
        assertTrue(Closeable.class.isAssignableFrom(OnnxFastenerDetector.class));
        assertEquals(boolean.class,
                OnnxFastenerDetector.class.getMethod("isReady").getReturnType());
        assertEquals(String.class,
                OnnxFastenerDetector.class.getMethod("getInitializationError").getReturnType());
        assertEquals(OnnxFastenerDetector.DetectionResult.class,
                OnnxFastenerDetector.class.getMethod("detect", Bitmap.class).getReturnType());
    }

    @Test
    public void retainsOneReusableInferenceWorkspace() throws Exception {
        assertFinalInstanceField("inputWorkspace", FastenerInputWorkspace.class);
    }

    @Test
    public void pinsOnnxRuntimeToTheSherpaCompatibleVersion() throws Exception {
        Path projectRoot = Paths.get(System.getProperty("user.dir"));
        Path buildFile = projectRoot.resolve("app").resolve("build.gradle");
        if (!Files.exists(buildFile)) {
            buildFile = projectRoot.resolve("build.gradle");
        }
        String buildScript = new String(
                Files.readAllBytes(buildFile), StandardCharsets.UTF_8);

        assertTrue(buildScript.contains(
                "com.microsoft.onnxruntime:onnxruntime-android:1.19.2"));
        assertTrue(!buildScript.contains(
                "com.microsoft.onnxruntime:onnxruntime-android:1.17.3"));
    }

    @Test(expected = UnsupportedOperationException.class)
    public void detectionResultDefensivelyCopiesItsDetections() {
        LetterboxTransform transform = LetterboxTransform.forSquare(640, 480, 640);
        List<Detection> detections = new ArrayList<>();
        detections.add(new Detection(1f, 2f, 3f, 4f, 0.9f, 0));
        OnnxFastenerDetector.DetectionResult result =
                new OnnxFastenerDetector.DetectionResult(
                        detections, 640, 480, 12.5, 2.0, 9.5, 1.0, transform);

        detections.clear();
        assertEquals(1, result.getDetections().size());
        assertEquals(640, result.getOriginalWidth());
        assertEquals(480, result.getOriginalHeight());
        assertEquals(12.5, result.getLatencyMillis(), 0.0001);
        assertEquals(2.0, result.getPreprocessMillis(), 0.0001);
        assertEquals(9.5, result.getInferenceMillis(), 0.0001);
        assertEquals(1.0, result.getPostprocessMillis(), 0.0001);
        assertEquals(transform, result.getTransform());
        result.getDetections().clear();
    }

    @Test
    public void calculatesLandscapeLetterboxGeometry() {
        LetterboxTransform transform = LetterboxTransform.forSquare(1280, 720, 640);

        assertEquals(1280, transform.getOriginalWidth());
        assertEquals(720, transform.getOriginalHeight());
        assertEquals(640, transform.getTargetWidth());
        assertEquals(640, transform.getTargetHeight());
        assertEquals(640, transform.getResizedWidth());
        assertEquals(360, transform.getResizedHeight());
        assertEquals(0, transform.getPadLeft());
        assertEquals(0, transform.getPadRight());
        assertEquals(140, transform.getPadTop());
        assertEquals(140, transform.getPadBottom());
        assertEquals(0.5f, transform.getScale(), EPSILON);
    }

    @Test
    public void calculatesPortraitLetterboxGeometry() {
        LetterboxTransform transform = LetterboxTransform.forSquare(720, 1280, 640);

        assertEquals(360, transform.getResizedWidth());
        assertEquals(640, transform.getResizedHeight());
        assertEquals(140, transform.getPadLeft());
        assertEquals(140, transform.getPadRight());
        assertEquals(0, transform.getPadTop());
        assertEquals(0, transform.getPadBottom());
        assertEquals(0.5f, transform.getScale(), EPSILON);
    }

    @Test
    public void assignsOddPaddingWithoutChangingTargetSize() {
        LetterboxTransform transform = LetterboxTransform.forSquare(641, 480, 640);

        assertEquals(640, transform.getResizedWidth());
        assertEquals(479, transform.getResizedHeight());
        assertEquals(80, transform.getPadTop());
        assertEquals(81, transform.getPadBottom());
        assertEquals(640,
                transform.getResizedWidth() + transform.getPadLeft() + transform.getPadRight());
        assertEquals(640,
                transform.getResizedHeight() + transform.getPadTop() + transform.getPadBottom());
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsNonPositiveSourceDimensions() {
        LetterboxTransform.forSquare(0, 480, 640);
    }

    @Test
    public void acceptsTheFrozenTensorShapes() {
        OnnxFastenerDetector.validateModelShapes(
                new long[]{1, 3, 640, 640},
                new long[]{1, 6, 34_000});
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsMissingTensorShapeMetadata() {
        OnnxFastenerDetector.validateModelShapes(null, new long[]{1, 6, 34_000});
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsUnexpectedInputShape() {
        OnnxFastenerDetector.validateModelShapes(
                new long[]{1, 3, 512, 512},
                new long[]{1, 6, 34_000});
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsUnexpectedOutputShape() {
        OnnxFastenerDetector.validateModelShapes(
                new long[]{1, 3, 640, 640},
                new long[]{1, 34_000, 6});
    }

    private static void assertFinalInstanceField(String name, Class<?> type) throws Exception {
        Field field = OnnxFastenerDetector.class.getDeclaredField(name);
        assertEquals(type, field.getType());
        assertTrue(Modifier.isFinal(field.getModifiers()));
        assertTrue(!Modifier.isStatic(field.getModifiers()));
    }
}
