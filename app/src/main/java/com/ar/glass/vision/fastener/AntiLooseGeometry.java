package com.ar.glass.vision.fastener;

public final class AntiLooseGeometry {
    private static final float MINIMUM_KEYPOINT_CONFIDENCE = 0.60f;
    private static final double EPSILON = 1.0e-6;

    private AntiLooseGeometry() {}

    /**
     * Legacy endpoint-only API. It cannot establish topology or mark ownership,
     * so it is deliberately fail-closed.
     */
    @Deprecated
    public static GeometryDecision evaluate(
            VisionPoint firstStart,
            VisionPoint firstEnd,
            VisionPoint secondStart,
            VisionPoint secondEnd,
            float referenceSize,
            GeometryThresholds thresholds) {
        return uncertain("TOPOLOGY_AND_MARK_ROLE_REQUIRED");
    }

    public static GeometryDecision evaluate(
            FastenerTopology topology,
            WitnessMarkRole markRole,
            boolean qualityPass,
            VisionPoint firstStart,
            VisionPoint firstEnd,
            VisionPoint secondStart,
            VisionPoint secondEnd,
            float referenceSize,
            GeometryThresholds thresholds) {
        if (topology == null || topology == FastenerTopology.UNKNOWN) {
            return uncertain("TOPOLOGY_UNKNOWN");
        }
        if (topology != FastenerTopology.BOLT_HEAD_PLATE
                && topology != FastenerTopology.NUT_PLATE) {
            return uncertain("TOPOLOGY_SOLVER_UNAVAILABLE");
        }
        if (markRole != WitnessMarkRole.BRIDGES_MOVING_FIXED) {
            return uncertain("MARK_DOES_NOT_BRIDGE_MOVING_FIXED");
        }
        if (!qualityPass) {
            return uncertain("IMAGE_QUALITY_FAILED");
        }
        VisionPoint[] points = {firstStart, firstEnd, secondStart, secondEnd};
        for (VisionPoint point : points) {
            if (point == null || !isFinite(point.getX()) || !isFinite(point.getY())
                    || !isFinite(point.getConfidence())) {
                return uncertain("KEYPOINT_VALUE_INVALID");
            }
            if (point.getConfidence() < MINIMUM_KEYPOINT_CONFIDENCE
                    || point.getConfidence() > 1f) {
                return uncertain("KEYPOINT_CONFIDENCE_LOW");
            }
        }
        if (thresholds == null || !thresholds.isCalibrated()) {
            return uncertain("THRESHOLDS_UNCALIBRATED");
        }
        if (!(referenceSize > 0f) || Float.isInfinite(referenceSize)) {
            return uncertain("REFERENCE_SIZE_INVALID");
        }

        double firstX = (double) firstEnd.getX() - firstStart.getX();
        double firstY = (double) firstEnd.getY() - firstStart.getY();
        double secondX = (double) secondEnd.getX() - secondStart.getX();
        double secondY = (double) secondEnd.getY() - secondStart.getY();
        double firstLength = Math.hypot(firstX, firstY);
        double secondLength = Math.hypot(secondX, secondY);
        if (firstLength < EPSILON || secondLength < EPSILON) {
            return uncertain("SEGMENT_LENGTH_INVALID");
        }

        double cosine = Math.abs((firstX * secondX + firstY * secondY) / (firstLength * secondLength));
        cosine = Math.max(-1.0, Math.min(1.0, cosine));
        double angleValue = Math.toDegrees(Math.acos(cosine));
        double gapValue = minimumEndpointDistance(firstStart, firstEnd, secondStart, secondEnd) / referenceSize;
        double residualValue = lineResidual(points) / referenceSize;
        if (!isFinite(angleValue) || !isFinite(gapValue) || !isFinite(residualValue)
                || angleValue > Float.MAX_VALUE || gapValue > Float.MAX_VALUE
                || residualValue > Float.MAX_VALUE) {
            return uncertain("GEOMETRY_VALUE_INVALID");
        }
        float angle = (float) angleValue;
        float gapRatio = (float) gapValue;
        float residualRatio = (float) residualValue;

        if (angle > thresholds.getMaximumAngleDegrees()) {
            return new GeometryDecision(FastenerState.INSUFFICIENT, "POSSIBLE_DISPLACED_ANGLE_EXCEEDED", angle, gapRatio, residualRatio);
        }
        if (gapRatio > thresholds.getMaximumGapRatio()) {
            return new GeometryDecision(FastenerState.INSUFFICIENT, "POSSIBLE_DISPLACED_GAP_EXCEEDED", angle, gapRatio, residualRatio);
        }
        if (residualRatio > thresholds.getMaximumResidualRatio()) {
            return new GeometryDecision(FastenerState.INSUFFICIENT, "POSSIBLE_DISPLACED_RESIDUAL_EXCEEDED", angle, gapRatio, residualRatio);
        }
        return new GeometryDecision(
                FastenerState.ALIGNED, "GEOMETRY_WITHIN_THRESHOLDS", angle, gapRatio, residualRatio);
    }

    private static GeometryDecision uncertain(String reason) {
        return new GeometryDecision(FastenerState.INSUFFICIENT, reason, Float.NaN, Float.NaN, Float.NaN);
    }

    private static boolean isFinite(float value) {
        return !Float.isNaN(value) && !Float.isInfinite(value);
    }

    private static boolean isFinite(double value) {
        return !Double.isNaN(value) && !Double.isInfinite(value);
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
