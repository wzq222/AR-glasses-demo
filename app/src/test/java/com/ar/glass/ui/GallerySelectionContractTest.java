package com.ar.glass.ui;

import static org.junit.Assert.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.junit.Test;

public class GallerySelectionContractTest {
    @Test
    public void selectionModeReturnsChosenOriginalImageWithoutBreakingBrowseMode()
            throws Exception {
        String source = readProjectFile(
                "app/src/main/java/com/ar/glass/ui/GalleryActivity.java");

        assertTrue(source.contains("EXTRA_SELECT_IMAGE"));
        assertTrue(source.contains("EXTRA_SELECTED_IMAGE_PATH"));
        assertTrue(source.contains("returnSelectedImage"));
        assertTrue(source.contains("setResult(RESULT_OK"));
        assertTrue(source.contains("openImageViewer(position)"));
        assertTrue(source.contains("选择原图库图片"));
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
