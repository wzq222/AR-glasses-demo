package com.ar.glass.vision.fastener;

public enum FastenerState {
    /** Witness mark is judgeable and no relative displacement was found. */
    ALIGNED,
    /** Independent evidence corroborates relative movement across the joint. */
    DISPLACED,
    /** Paint is damaged or degraded without evidence of rigid-body movement. */
    DAMAGED_MARK,
    /** Image, topology, segment binding, calibration, or evidence is insufficient. */
    INSUFFICIENT,
    /** Legacy wire value; new analyzers never return it. */
    @Deprecated NORMAL,
    /** Legacy wire value; new analyzers never return it. */
    @Deprecated LOOSE,
    /** Legacy wire value; new analyzers never return it. */
    @Deprecated UNCERTAIN
}
