# High-recall threshold recovery and phone gate

## Outcome

The default Android ncnn confidence threshold had regressed to `0.20`, while the frozen
marked-point recall gate was calibrated at `0.0019424824276939034`. The default has been
restored to the calibrated value and guarded by an Android contract test.

This model detects inspection points carrying an anti-loosening mark. It is not an
all-physical-bolt detector, and the result below must not be presented as cross-vehicle or
production accuracy.

## Physical-device verification

- Device: Huawei `NOH-AN01`, serial `7TD0221529000027`.
- Run token: `recall-gate-r3`.
- Input: the frozen 17-image `marked-point-v1.4` development validation set.
- Run contract: 17 image summaries, 294 detailed phone boxes, one 17-image completion
  marker, and the detailed-box count exactly matching the summary count.
- Legacy permissive coverage: 17/17 complete scenes and 75/75 truth objects. This used
  candidate-center/low-IoU/containment matching and is not a localization result.
- Corrected one-to-one localization audit: 62/75 at IoU 0.30 (8/17 complete scenes),
  and 40/75 at IoU 0.50 (3/17 complete scenes). The current mobile model therefore
  fails the strict localization gate.
- Total-latency P50: 732.355 ms/image. One 4.071 s outlier occurred, so this short run is
  an accuracy gate rather than a stable thermal latency claim.
- Formal fastener truth SHA-256 remained
  `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`.

The debug APK installed for this run has SHA-256
`25E2387A1CE1812C5B1F92E76EFA5281D9B8734026ADF8177388276DA9F96196`.

## Overlapping global/tile candidates

The separate all-fastener challenge on `LOCK-REAL-01.jpg` showed that ordinary IoU NMS
does not remove every duplicate after global and overlapping-tile coordinates are mapped
back to the source image. Raising the detector threshold is rejected because recall is the
primary requirement.

A Git-external source-aware fusion prototype now clusters only cross-view candidates using
IoU, containment, normalized center distance, area ratio, and aspect-ratio gates. It keeps
the highest-score box as the visible target and stores the union as a larger review ROI.
Four initial regression tests cover cross-view merging, same-view preservation, large-box
swallowing, and adjacent-object preservation. On the supplied challenge report it reduced
52 candidates to 46 by merging six evidence-backed duplicates without score filtering.

A follow-up display-only filter hides broad context boxes that contain a much tighter
candidate and deterministic duplicates while retaining all original review evidence. It
reduces this image from 52 raw candidates to 41 displayed boxes. This improves readability,
but the remaining false positives prove that post-processing alone cannot repair the model.

This prototype is not yet in the Android inference path. It needs regression over complete
scenes with adjacent fasteners before integration; otherwise a large false box could merge
two true neighbouring objects and reduce recall.
