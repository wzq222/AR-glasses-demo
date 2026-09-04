package com.ar.glass.sop;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

public class SopEvidenceFilesTest {
    @Rule public TemporaryFolder temporaryFolder = new TemporaryFolder();

    @Test
    public void copiesSupportedImageIntoUniqueEvidenceFile() throws Exception {
        File first = SopEvidenceFiles.copy(
                new ByteArrayInputStream(new byte[]{1, 2, 3}),
                "image/png", temporaryFolder.getRoot(), 1234L);
        File second = SopEvidenceFiles.copy(
                new ByteArrayInputStream(new byte[]{4, 5}),
                "image/png", temporaryFolder.getRoot(), 1234L);

        assertEquals("sop_1234.png", first.getName());
        assertEquals("sop_1234_1.png", second.getName());
        assertArrayEquals(new byte[]{1, 2, 3}, Files.readAllBytes(first.toPath()));
        assertEquals("image/png", SopEvidenceFiles.mediaType(first));
    }

    @Test
    public void rejectsUnsupportedAndEmptyInputWithoutLeavingPartialFile() throws Exception {
        assertThrows(IllegalArgumentException.class,
                () -> SopEvidenceFiles.extensionFor("video/mp4"));
        assertThrows(IOException.class, () -> SopEvidenceFiles.copy(
                new ByteArrayInputStream(new byte[0]),
                "image/jpeg", temporaryFolder.getRoot(), 1L));
        assertEquals(0, temporaryFolder.getRoot().listFiles().length);
    }

    @Test
    public void rejectsMoreThanTwentyFiveMibAndDeletesPartialFile() throws Exception {
        assertThrows(IOException.class, () -> SopEvidenceFiles.copy(
                new SizedInputStream(SopEvidenceFiles.MAX_BYTES + 1),
                "image/webp", temporaryFolder.getRoot(), 9L));

        File partial = new File(temporaryFolder.getRoot(), "sop_9.webp");
        assertFalse(partial.exists());
    }

    private static final class SizedInputStream extends InputStream {
        private long remaining;

        SizedInputStream(long remaining) {
            this.remaining = remaining;
        }

        @Override public int read() {
            if (remaining <= 0) return -1;
            remaining--;
            return 0;
        }

        @Override public int read(byte[] buffer, int offset, int length) {
            if (remaining <= 0) return -1;
            int count = (int) Math.min(length, remaining);
            remaining -= count;
            return count;
        }
    }
}
