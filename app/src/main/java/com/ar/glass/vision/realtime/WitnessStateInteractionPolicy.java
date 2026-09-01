package com.ar.glass.vision.realtime;

public final class WitnessStateInteractionPolicy {
    public enum Availability {
        DISABLED,
        UNAVAILABLE,
        READY
    }

    private final Availability availability;

    private WitnessStateInteractionPolicy(Availability availability) {
        this.availability = availability;
    }

    public static WitnessStateInteractionPolicy forRuntime(
            boolean experimentalBuild, boolean estimatorReady) {
        if (!experimentalBuild) {
            return new WitnessStateInteractionPolicy(Availability.DISABLED);
        }
        return new WitnessStateInteractionPolicy(
                estimatorReady ? Availability.READY : Availability.UNAVAILABLE);
    }

    public Availability getAvailability() {
        return availability;
    }

    public boolean canTapCandidate() {
        return availability == Availability.READY;
    }
}
