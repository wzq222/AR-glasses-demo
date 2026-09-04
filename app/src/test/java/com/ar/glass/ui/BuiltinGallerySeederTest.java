package com.ar.glass.ui;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

public class BuiltinGallerySeederTest {
    @Rule public TemporaryFolder temporaryFolder = new TemporaryFolder();

    @Test
    public void recursivelyCopiesSupportedImagesAndIsIdempotent() throws Exception {
        MapSource source = new MapSource()
                .put("qr/QR-REAL-01.jpg", new byte[]{1, 2, 3})
                .put("fastener/LOCK-REAL-01.jpg", new byte[]{4, 5})
                .put("meter/METER-04_zero_000.jpg", new byte[]{6})
                .put("meter/manifest.csv", new byte[]{7, 8});
        File target = temporaryFolder.newFolder("builtin");

        assertEquals(3, BuiltinGallerySeeder.seed(source, target));
        assertArrayEquals(new byte[]{1, 2, 3}, Files.readAllBytes(
                new File(target, "qr/QR-REAL-01.jpg").toPath()));
        assertArrayEquals(new byte[]{4, 5}, Files.readAllBytes(
                new File(target, "fastener/LOCK-REAL-01.jpg").toPath()));
        assertArrayEquals(new byte[]{6}, Files.readAllBytes(
                new File(target, "meter/METER-04_zero_000.jpg").toPath()));
        assertFalse(new File(target, "meter/manifest.csv").exists());
        assertEquals(0, BuiltinGallerySeeder.seed(source, target));
    }

    @Test
    public void rejectsUnsafeAssetSegmentsWithoutWritingOutsideRoot() throws Exception {
        BuiltinGallerySeeder.Source source = new BuiltinGallerySeeder.Source() {
            @Override public String[] list(String relativePath) {
                return relativePath.isEmpty() ? new String[]{".."} : new String[0];
            }

            @Override public InputStream open(String relativePath) {
                return new ByteArrayInputStream(new byte[]{9});
            }
        };
        File target = temporaryFolder.newFolder("safe");

        assertThrows(IOException.class, () -> BuiltinGallerySeeder.seed(source, target));
        assertEquals(0, target.listFiles().length);
        assertFalse(new File(target.getParentFile(), "outside.jpg").exists());
    }

    @Test
    public void replacesInterruptedPartFileOnlyAfterCompleteCopy() throws Exception {
        MapSource source = new MapSource().put("qr/example.webp", new byte[]{3, 2, 1});
        File target = temporaryFolder.newFolder("atomic");
        File directory = new File(target, "qr");
        assertTrue(directory.mkdirs());
        File partial = new File(directory, "example.webp.part");
        Files.write(partial.toPath(), new byte[]{0});

        assertEquals(1, BuiltinGallerySeeder.seed(source, target));
        assertFalse(partial.exists());
        assertArrayEquals(new byte[]{3, 2, 1}, Files.readAllBytes(
                new File(directory, "example.webp").toPath()));
    }

    private static final class MapSource implements BuiltinGallerySeeder.Source {
        private final Map<String, byte[]> files = new LinkedHashMap<>();

        MapSource put(String path, byte[] bytes) {
            files.put(path, bytes);
            return this;
        }

        @Override public String[] list(String relativePath) {
            String prefix = relativePath.isEmpty() ? "" : relativePath + "/";
            Set<String> children = new LinkedHashSet<>();
            for (String path : files.keySet()) {
                if (!path.startsWith(prefix)) continue;
                String remainder = path.substring(prefix.length());
                int slash = remainder.indexOf('/');
                children.add(slash >= 0 ? remainder.substring(0, slash) : remainder);
            }
            List<String> ordered = new ArrayList<>(children);
            return ordered.toArray(new String[0]);
        }

        @Override public InputStream open(String relativePath) throws IOException {
            byte[] bytes = files.get(relativePath);
            if (bytes == null) throw new IOException("not a file: " + relativePath);
            return new ByteArrayInputStream(bytes);
        }
    }
}
