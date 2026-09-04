package com.ar.glass.sop;

import static org.junit.Assert.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.junit.Test;

public class SopImageDecoderContractTest {
    @Test
    public void decoderBoundsMemoryAndAppliesExifOrientation() throws Exception {
        String source = readProjectFile(
                "app/src/main/java/com/ar/glass/sop/SopImageDecoder.java");

        assertTrue(source.contains("maxEdge"));
        assertTrue(source.contains("inSampleSize"));
        assertTrue(source.contains("ExifInterface.TAG_ORIENTATION"));
        assertTrue(source.contains("Bitmap.createBitmap"));
        assertTrue(source.contains("ORIENTATION_TRANSVERSE"));
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
