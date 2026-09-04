package com.ar.glass.sop;

import static org.junit.Assert.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.junit.Test;

public class SopSubmittedAssignmentContractTest {
    @Test
    public void submittedAssignmentsAreNotShownAsMobileWork() throws Exception {
        Path root = Paths.get(System.getProperty("user.dir"));
        Path file = root.resolve("app/src/main/java/com/ar/glass/sop/SopActivity.java");
        if (!Files.exists(file)) {
            file = root.resolve("src/main/java/com/ar/glass/sop/SopActivity.java");
        }
        String source = new String(Files.readAllBytes(file), StandardCharsets.UTF_8);
        assertTrue(source.contains("!\"submitted\".equals(item.status)"));
    }
}
