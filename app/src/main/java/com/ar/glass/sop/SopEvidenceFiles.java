package com.ar.glass.sop;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Locale;

public final class SopEvidenceFiles {
    public static final long MAX_BYTES = 25L * 1024L * 1024L;

    private SopEvidenceFiles() {}

    public static String extensionFor(String mime) {
        String normalized = mime == null ? "" : mime.trim().toLowerCase(Locale.ROOT);
        if ("image/jpeg".equals(normalized) || "image/jpg".equals(normalized)) return ".jpg";
        if ("image/png".equals(normalized)) return ".png";
        if ("image/webp".equals(normalized)) return ".webp";
        throw new IllegalArgumentException("仅支持 JPEG、PNG 或 WebP 图片");
    }

    public static File copy(InputStream input, String mime, File directory, long now)
            throws IOException {
        if (input == null) throw new IOException("无法读取所选图片");
        if (directory == null) throw new IOException("证据目录不可用");
        if (!directory.exists() && !directory.mkdirs()) {
            throw new IOException("无法创建证据目录");
        }
        if (!directory.isDirectory()) throw new IOException("证据目录不可用");

        File output = unique(directory, "sop_" + now, extensionFor(mime));
        long total = 0L;
        boolean complete = false;
        try (InputStream source = input;
             FileOutputStream target = new FileOutputStream(output)) {
            byte[] buffer = new byte[8192];
            int count;
            while ((count = source.read(buffer)) != -1) {
                total += count;
                if (total > MAX_BYTES) throw new IOException("图片超过25 MiB");
                target.write(buffer, 0, count);
            }
            if (total == 0L) throw new IOException("所选图片为空");
            target.flush();
            complete = true;
            return output;
        } finally {
            if (!complete && output.exists()) output.delete();
        }
    }

    public static String mediaType(File file) {
        if (file == null) throw new IllegalArgumentException("证据文件为空");
        String name = file.getName().toLowerCase(Locale.ROOT);
        if (name.endsWith(".jpg") || name.endsWith(".jpeg")) return "image/jpeg";
        if (name.endsWith(".png")) return "image/png";
        if (name.endsWith(".webp")) return "image/webp";
        throw new IllegalArgumentException("不支持的证据图片类型");
    }

    private static File unique(File directory, String base, String extension) {
        File candidate = new File(directory, base + extension);
        for (int suffix = 1; candidate.exists(); suffix++) {
            candidate = new File(directory, base + "_" + suffix + extension);
        }
        return candidate;
    }
}
