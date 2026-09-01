package com.ar.glass.ui;

import android.widget.Button;

import com.ar.glass.R;

import org.junit.Test;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

public class MainActivityLiveInspectionContractTest {
    @Test
    public void exposesTheLiveInspectionEntryAndCopyResources() {
        assertNotEquals(0, R.id.btnLiveInspection);
        assertNotEquals(0, R.string.live_inspection_entry_title);
        assertNotEquals(0, R.string.live_inspection_entry_disclaimer);
        assertNotEquals(0, R.string.live_back);
        assertNotEquals(0, R.string.live_model_loading);
        assertNotEquals(0, R.string.live_metrics_placeholder);
        assertNotEquals(0, R.string.live_safety_refusal);
        assertNotEquals(0, R.string.live_camera_permission_denied);
        assertNotEquals(0, R.string.live_camera_start_failed);
        assertNotEquals(0, R.string.live_model_ready);
        assertNotEquals(0, R.string.live_model_missing);
        assertNotEquals(0, R.string.live_model_initialization_error);
        assertNotEquals(0, R.string.live_frame_inference_failed);
        assertNotEquals(0, R.string.live_metrics_format);
        assertNotEquals(0, R.string.live_state_experimental_title);
        assertNotEquals(0, R.string.live_state_experimental_warning);
        assertNotEquals(0, R.string.live_state_unavailable);
        assertNotEquals(0, R.string.live_state_result_format);
        assertNotEquals(0, R.string.live_safety_experimental_enabled);
        assertNotEquals(0, R.string.live_safety_experimental_unavailable);
    }

    @Test
    public void defaultCopyRetainsRefusalAndSeparatesExperimentalAvailability() throws Exception {
        Path projectRoot = Paths.get(System.getProperty("user.dir"));
        Path strings = projectRoot.resolve("app/src/main/res/values/strings.xml");
        if (!Files.exists(strings)) {
            strings = projectRoot.resolve("src/main/res/values/strings.xml");
        }
        String text = new String(Files.readAllBytes(strings), StandardCharsets.UTF_8);

        assertTrue(text.contains(
                "<string name=\"live_inspection_entry_disclaimer\">"
                        + "防松标记候选检测；松动状态暂需人工确认</string>"));
        assertTrue(text.contains(
                "<string name=\"live_safety_refusal\">"
                        + "松动状态：无法判断（实验状态模型未启用）</string>"));
        assertTrue(text.contains("name=\"live_safety_experimental_enabled\""));
        assertTrue(text.contains("name=\"live_safety_experimental_unavailable\""));
    }

    @Test
    public void mainActivityOwnsAnExplicitLiveInspectionLauncher() throws Exception {
        Field button = MainActivity.class.getDeclaredField("btnLiveInspection");
        assertEquals(Button.class, button.getType());
        assertTrue(Modifier.isPrivate(button.getModifiers()));

        Method launcher = MainActivity.class.getDeclaredMethod("openLiveInspection");
        assertEquals(void.class, launcher.getReturnType());
        assertTrue(Modifier.isPrivate(launcher.getModifiers()));
        assertEquals(0, launcher.getParameterTypes().length);
    }

    @Test
    public void bleControlGateDoesNotOwnTheLiveInspectionButton() throws Exception {
        String source = readMainActivitySource();
        String controlsBody = methodBody(source, "private void setControlsEnabled(boolean enabled)");

        assertFalse(controlsBody.contains("btnLiveInspection"));
        assertTrue(source.contains("btnLiveInspection.setOnClickListener(v -> openLiveInspection())"));
        assertTrue(source.contains("new Intent(this, LiveInspectionActivity.class)"));
    }

    private static String readMainActivitySource() throws Exception {
        Path projectRoot = Paths.get(System.getProperty("user.dir"));
        Path source = projectRoot.resolve("app/src/main/java/com/ar/glass/ui/MainActivity.java");
        if (!Files.exists(source)) {
            source = projectRoot.resolve("src/main/java/com/ar/glass/ui/MainActivity.java");
        }
        return new String(Files.readAllBytes(source), StandardCharsets.UTF_8);
    }

    private static String methodBody(String source, String signature) {
        int signatureStart = source.indexOf(signature);
        assertTrue("Missing method: " + signature, signatureStart >= 0);
        int bodyStart = source.indexOf('{', signatureStart);
        int depth = 0;
        for (int index = bodyStart; index < source.length(); index++) {
            char character = source.charAt(index);
            if (character == '{') {
                depth++;
            } else if (character == '}') {
                depth--;
                if (depth == 0) {
                    return source.substring(bodyStart, index + 1);
                }
            }
        }
        throw new AssertionError("Unterminated method: " + signature);
    }
}
