package com.ar.glass.ui;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Locale;

/** Copies APK-bundled case images into a stable app-private directory. */
public final class BuiltinGallerySeeder {
    private static final int BUFFER_SIZE = 8192;

    public interface Source {
        String[] list(String relativePath) throws IOException;
        InputStream open(String relativePath) throws IOException;
    }

    private BuiltinGallerySeeder() {}

    public static int seed(Source source, File targetRoot) throws IOException {
        if (source == null || targetRoot == null) {
            throw new IllegalArgumentException("source and targetRoot are required");
        }
        if (!targetRoot.exists() && !targetRoot.mkdirs()) {
            throw new IOException("cannot create built-in gallery directory");
        }
        return seedDirectory(source, targetRoot.getCanonicalFile(), "");
    }

    private static int seedDirectory(Source source, File targetRoot, String relativePath)
            throws IOException {
        String[] children = source.list(relativePath);
        if (children == null || children.length == 0) return 0;

        int copied = 0;
        for (String child : children) {
            validateSegment(child);
            String childPath = relativePath.isEmpty() ? child : relativePath + "/" + child;
            String[] grandchildren = source.list(childPath);
            if (grandchildren != null && grandchildren.length > 0) {
                copied += seedDirectory(source, targetRoot, childPath);
            } else if (isSupportedImage(child)) {
                copied += copyFile(source, targetRoot, childPath);
            }
        }
        return copied;
    }

    private static int copyFile(Source source, File targetRoot, String relativePath)
            throws IOException {
        File target = new File(targetRoot, relativePath.replace('/', File.separatorChar))
                .getCanonicalFile();
        String rootPath = targetRoot.getPath() + File.separator;
        if (!target.getPath().startsWith(rootPath)) {
            throw new IOException("unsafe built-in gallery path");
        }
        File parent = target.getParentFile();
        if (!parent.exists() && !parent.mkdirs()) {
            throw new IOException("cannot create built-in gallery subdirectory");
        }

        File partial = new File(parent, target.getName() + ".part");
        if (partial.exists() && !partial.delete()) {
            throw new IOException("cannot clear interrupted built-in gallery copy");
        }

        long bytes = 0L;
        try (InputStream input = source.open(relativePath);
             FileOutputStream output = new FileOutputStream(partial)) {
            byte[] buffer = new byte[BUFFER_SIZE];
            int count;
            while ((count = input.read(buffer)) != -1) {
                output.write(buffer, 0, count);
                bytes += count;
            }
            output.flush();
        } catch (IOException error) {
            partial.delete();
            throw error;
        }

        if (bytes <= 0L) {
            partial.delete();
            throw new IOException("empty built-in gallery image");
        }
        if (target.isFile() && target.length() == bytes) {
            partial.delete();
            return 0;
        }
        if (target.exists() && !target.delete()) {
            partial.delete();
            throw new IOException("cannot replace built-in gallery image");
        }
        if (!partial.renameTo(target)) {
            partial.delete();
            throw new IOException("cannot finalize built-in gallery image");
        }
        return 1;
    }

    private static boolean isSupportedImage(String name) {
        String lower = name.toLowerCase(Locale.US);
        return lower.endsWith(".jpg") || lower.endsWith(".jpeg")
                || lower.endsWith(".png") || lower.endsWith(".webp");
    }

    private static void validateSegment(String segment) throws IOException {
        if (segment == null || segment.isEmpty() || ".".equals(segment)
                || "..".equals(segment) || segment.indexOf('/') >= 0
                || segment.indexOf('\\') >= 0 || new File(segment).isAbsolute()) {
            throw new IOException("unsafe built-in gallery asset segment");
        }
    }
}
