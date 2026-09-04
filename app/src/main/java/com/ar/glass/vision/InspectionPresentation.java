package com.ar.glass.vision;

import com.ar.glass.vision.realtime.WitnessStateEstimate;
import com.ar.glass.vision.realtime.WitnessTriage;

import java.util.Locale;

/** User-facing two-stage inspection labels: locate the bolt, then assess looseness evidence. */
public final class InspectionPresentation {
    private InspectionPresentation() {}

    public static String boltLabel(int index) {
        return "螺栓 " + index + "（有防松标记）";
    }

    public static String stateLabel(WitnessStateEstimate estimate) {
        if (estimate == null) return "无法判断 · 请近拍审核";
        return stateLabel(estimate.getTriage().name(), estimate.getAngleDegrees());
    }

    public static String stateLabel(String triage, float angleDegrees) {
        if (WitnessTriage.LIKELY_ALIGNED.name().equals(triage) && Float.isFinite(angleDegrees)) {
            return angleLabel("未见松动迹象", angleDegrees, false);
        }
        if (WitnessTriage.POSSIBLE_DISPLACED.name().equals(triage)
                && Float.isFinite(angleDegrees)) {
            return angleLabel("疑似松动", angleDegrees, true);
        }
        if (WitnessTriage.HIGH_SUSPICION.name().equals(triage)
                && Float.isFinite(angleDegrees)) {
            return angleLabel("高疑似松动", angleDegrees, true);
        }
        return "无法判断 · 请近拍审核";
    }

    private static String angleLabel(String state, float angleDegrees, boolean review) {
        return String.format(
                Locale.US,
                review ? "%s · 夹角 %.1f° · 请审核" : "%s · 夹角 %.1f°",
                state,
                angleDegrees);
    }
}
