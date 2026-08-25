package com.ar.glass.vision.fastener;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class FullImageInspectionResult {
    private final int imageWidth;
    private final int imageHeight;
    private final long elapsedMillis;
    private final List<FastenerInspection> fasteners;

    public FullImageInspectionResult(
            int imageWidth,
            int imageHeight,
            long elapsedMillis,
            List<FastenerInspection> fasteners) {
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        this.elapsedMillis = elapsedMillis;
        this.fasteners = Collections.unmodifiableList(new ArrayList<>(fasteners));
    }

    public int getImageWidth() { return imageWidth; }
    public int getImageHeight() { return imageHeight; }
    public long getElapsedMillis() { return elapsedMillis; }
    public List<FastenerInspection> getFasteners() { return fasteners; }
}
