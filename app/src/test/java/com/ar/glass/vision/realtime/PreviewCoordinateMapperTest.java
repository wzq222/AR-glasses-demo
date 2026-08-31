package com.ar.glass.vision.realtime;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class PreviewCoordinateMapperTest {
    private static final float EPSILON = 0.001f;

    @Test
    public void fillCenterMapsLandscapeFrameWithHorizontalCrop() {
        PreviewCoordinateMapper mapper = PreviewCoordinateMapper.fillCenter(
                640, 480, 1000, 1000);

        PreviewCoordinateMapper.MappedRect mapped = mapper.map(
                160f, 120f, 480f, 360f);

        assertEquals(166.667f, mapped.getLeft(), EPSILON);
        assertEquals(250f, mapped.getTop(), EPSILON);
        assertEquals(833.333f, mapped.getRight(), EPSILON);
        assertEquals(750f, mapped.getBottom(), EPSILON);
    }

    @Test
    public void fillCenterMapsPortraitFrameWithHorizontalCrop() {
        PreviewCoordinateMapper mapper = PreviewCoordinateMapper.fillCenter(
                480, 640, 1080, 1920);

        PreviewCoordinateMapper.MappedRect mapped = mapper.map(
                0f, 0f, 480f, 640f);

        assertEquals(-180f, mapped.getLeft(), EPSILON);
        assertEquals(0f, mapped.getTop(), EPSILON);
        assertEquals(1260f, mapped.getRight(), EPSILON);
        assertEquals(1920f, mapped.getBottom(), EPSILON);
    }
}
