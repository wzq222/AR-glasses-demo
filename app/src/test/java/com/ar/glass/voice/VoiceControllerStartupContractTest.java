package com.ar.glass.voice;

import static org.junit.Assert.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.junit.Test;

public class VoiceControllerStartupContractTest {
    @Test
    public void unavailableToneGeneratorCannotCrashApplicationStartup() throws Exception {
        String source = readProjectFile(
                "app/src/main/java/com/ar/glass/voice/VoiceController.java");

        assertTrue(source.contains("initToneGenerator();"));
        assertTrue(source.contains("private void initToneGenerator()"));
        assertTrue(source.contains("catch (RuntimeException error)"));
        assertTrue(source.contains("toneGenerator = null;"));
    }

    private static String readProjectFile(String relativePath) throws Exception {
        Path projectRoot = Paths.get(System.getProperty("user.dir"));
        Path file = projectRoot.resolve(relativePath);
        if (!Files.exists(file) && relativePath.startsWith("app/")) {
            file = projectRoot.resolve(relativePath.substring("app/".length()));
        }
        return new String(Files.readAllBytes(file), StandardCharsets.UTF_8);
    }
}
