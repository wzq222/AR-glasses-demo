package com.ar.glass.vision.fastener;

public final class AntiLooseGeometry {
    private static final float MINIMUM_KEYPOINT_CONFIDENCE = 0.60f;
    private static final double EPSILON = 1.0e-6;

    private AntiLooseGeometry() {}

    public static GeometryDecision evaluate(
            VisionPoint firstStart,
            VisionPoint firstEnd,
            VisionPoint secondStart,
            VisionPoint secondEnd,
            float referenceSize,
            GeometryThresholds thresholds) {
        VisionPoint[] points = {firstStart, firstEnd, secondStart, secondEnd};
        for (VisionPoint point : points) {
            if (point == null || point.getConfidence() < MINIMUM_KEYPOINT_CONFIDENCE) {
                return uncertain("KEYPOINT_CONFIDENCE_LOW");
            }
        }
        if (thresholds == null || !thresholds.isCalibrated()) {
            return uncertain("THRESHOLDS_UNCALIBRATED");
        }
        if (!(referenceSize > 0f) || Float.isInfinite(referenceSize)) {
            return uncertain("REFERENCE_SIZE_INVALID");
        }

        double firstX = firstEnd.getX() - firstStart.getX();
        double firstY = firstEnd.getY() - firstStart.getY();
        double secondX = secondEnd.getX() - secondStart.getX();
        double secondY = secondEnd.getY() - secondStart.getY();
        double firstLength = Math.hypot(firstX, firstY);
        double secondLength = Math.hypot(secondX, secondY);
        if (firstLength < EPSILON || secondLength < EPSILON) {
            return uncertain("SEGMENT_LENGTH_INVALID");
        }

        double cosine = Math.abs((firstX * secondX + firstY * secondY) / (firstLength * secondLength));
        cosine = Math.max(-1.0, Math.min(1.0, cosine));
        float angle = (float) Math.toDegrees(Math.acos(cosine));
        float gapRatio = (float) (minimumEndpointDistance(firstStart, firstEnd, secondStart, secondEnd) / referenceSize);
        float residualRatio = (float) (lineResidual(points) / referenceSize);

        if (angle > thresholds.getMaximumAngleDegrees()) {
            return new GeometryDecision(FastenerState.LOOSE, "ANGLE_EXCEEDED", angle, gapRatio, residualRatio);
        }
        if (gapRatio > thresholds.getMaximumGapRatio()) {
            return new GeometryDecision(FastenerState.LOOSE, "GAP_EXCEEDED", angle, gapRatio, residualRatio);
        }
        if (residualRatio > thresholds.getMaximumResidualRatio()) {
            return new GeometryDecision(FastenerState.LOOSE, "RESIDUAL_EXCEEDED", angle, gapRatio, residualRatio);
        }
        return new GeometryDecision(
                FastenerState.NORMAL, "GEOMETRY_WITHIN_THRESHOLDS", angle, gapRatio, residualRatio);
    }

    private static GeometryDecision uncertain(String reason) {
        return new GeometryDecision(FastenerState.UNCERTAIN, reason, Float.NaN, Float.NaN, Float.NaN);
    }

    private static double minimumEndpointDistance(
            VisionPoint a, VisionPoint b, VisionPoint c, VisionPoint d) {
        return Math.min(
                Math.min(distance(a, c), distance(a, d)),
                Math.min(distance(b, c), distance(b, d)));
    }

    private static double distance(VisionPoint left, VisionPoint right) {
        return Math.hypot(left.getX() - right.getX(), left.getY() - right.getY());
    }

    private static double lineResidual(VisionPoint[] points) {
        double centerX = 0.0;
        double centerY = 0.0;
        for (VisionPoint point : points) {
            centerX += point.getX();
            centerY += point.getY();
        }
        centerX /= points.length;
        centerY /= points.length;

        double xx = 0.0;
        double yy = 0.0;
        double xy = 0.0;
        for (VisionPoint point : points) {
            double x = point.getX() - centerX;
            double y = point.getY() - centerY;
            xx += x * x;
            yy += y * y;
            xy += x * y;
        }
        double angle = 0.5 * Math.atan2(2.0 * xy, xx - yy);
        double normalX = -Math.sin(angle);
        double normalY = Math.cos(angle);
        double squaredResidual = 0.0;
        for (VisionPoint point : points) {
            double projection = (point.getX() - centerX) * normalX + (point.getY() - centerY) * normalY;
            squaredResidual += projection * projection;
        }
        return Math.sqrt(squaredResidual / points.length);
    }
}
