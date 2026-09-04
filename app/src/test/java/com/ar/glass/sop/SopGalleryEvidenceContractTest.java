package com.ar.glass.sop;

import static org.junit.Assert.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.junit.Test;

public class SopGalleryEvidenceContractTest {
    @Test
    public void sopOffersThreeImageSourcesAndRecordsTheSelectedSource() throws Exception {
        String activity = read("app/src/main/java/com/ar/glass/sop/SopActivity.java");
        String layout = read("app/src/main/res/layout/activity_sop.xml");
        String api = read("app/src/main/java/com/ar/glass/sop/SopApiClient.java");

        assertTrue(layout.contains("btnSelectPhoneGallery"));
        assertTrue(layout.contains("btnSelectOriginalGallery"));
        assertTrue(layout.contains("从手机图库选择"));
        assertTrue(layout.contains("从 App 原图库选择"));
        assertTrue(activity.contains("ActivityResultContracts.GetContent"));
        assertTrue(activity.contains("GalleryActivity.EXTRA_SELECT_IMAGE"));
        assertTrue(activity.contains("pendingCameraFile"));
        assertTrue(activity.contains("processEvidenceFile"));
        assertTrue(activity.contains("value.put(\"evidence_source\", evidenceSource)"));
        assertTrue(api.contains("SopEvidenceFiles.mediaType(file)"));
    }

    private static String read(String path) throws Exception {
        Path projectRoot = Paths.get(System.getProperty("user.dir"));
        Path file = projectRoot.resolve(path);
        if (!Files.exists(file) && path.startsWith("app/")) {
            file = projectRoot.resolve(path.substring("app/".length()));
        }
        return new String(Files.readAllBytes(file), StandardCharsets.UTF_8);
    }
}
