package com.ar.glass.ui;

import static org.junit.Assert.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.junit.Test;

public class BundledGalleryBuildContractTest {
    @Test
    public void gradlePackagesOnlySupportedGitExternalCaseImages() throws Exception {
        String gradle = readProjectFile("app/build.gradle");

        assertTrue(gradle.contains("CRRC_BUILTIN_GALLERY_DIR"));
        assertTrue(gradle.contains("generated/builtinGalleryAssets"));
        assertTrue(gradle.contains("prepareBuiltinGalleryAssets"));
        assertTrue(gradle.contains("builtin_gallery"));
        assertTrue(gradle.contains("**/*.jpg"));
        assertTrue(gradle.contains("**/*.jpeg"));
        assertTrue(gradle.contains("**/*.png"));
        assertTrue(gradle.contains("**/*.webp"));
        assertTrue(gradle.contains("preBuild.dependsOn prepareBuiltinGalleryAssets"));
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
