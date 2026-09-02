# Witness-line SOP assistive integration

## Outcome

The phone SOP `FASTENER_MARK` step now runs the existing full-image marked-point
detector and verifier, then evaluates every retained inspection point with the
experimental witness-line ROI model. The result is deliberately an assistive triage:
the operator must confirm every point before the step can be saved.

This change does not promote the state model to a production `ALIGNED/DISPLACED`
classifier. The 19 historical real ROIs still produce 19/19 strict evidence-gate
rejections. Controlled real before/after pairs and an independent test set are still
required for an automatic looseness accuracy claim.

## Product behavior

- Expand every retained marked-point box to a 1.5x ROI and run `witness-roi.onnx`.
- Refuse state measurement for a target shorter than 32 px, unlocalized keypoint
  heatmaps, missing witness-mask support, or incoherent two-segment topology.
- For measurable evidence, calculate the angle between the fixed-side and moving-side
  witness segments and route it at 3 and 15 degrees.
- Draw the point box, the two predicted witness segments, and four endpoints over the
  full evidence photo.
- Require one operator choice per point: confirmed aligned, suspected displaced, or
  unable to judge/recapture.
- Store the AI triage, angle, experimental +/-6.3 degree interval, failure reason,
  bounding box, confidence, and operator decision in `point_results`.
- Derive the saved step `state` from the completed human review, while preserving the
  original AI triage separately.

## Frozen assets

| APK asset | Bytes | SHA-256 |
|---|---:|---|
| `fastener-target-p2-640.onnx` | 43,245,031 | `C50F9105FF75885BE3BA02464E6A994FA7A45FDE0B0634AEA12FAA04A6CC5B7A` |
| `marked-point-verifier.onnx` | 6,089,246 | `FED197A11134DD4358B70EFF64086C050DDECC9B2C484E72AAEB102E4BA563CD` |
| `witness-roi.onnx` | 5,458,941 | `6D42E0D6C5785866DC65077FCD4D5E6EED576689431CA5C3E6649A280A5880BA` |

All three APK entries were streamed from the built APK and matched their source asset
in both byte count and SHA-256.

## Verification

- `testDebugUnitTest`: 54 tests, 0 failures, 0 errors.
- `assembleDebug`: PASS.
- Debug APK: 101,382,296 bytes.
- Debug APK SHA-256:
  `2F6227AF1B9A852CAF85C76EAC4AECE1A855EB41A117A892EE5BF759190F67E9`.
- Formal truth `annotations/fastener-v2/instances.json` remained:
  `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`.
- ADB found no connected Android device during this build, so installation and camera
  interaction remain pending. No APK was installed on the glasses.

The first Android resource merge after a clean/short-path transition hit the known
legacy Gradle transient `!directory.isDirectory`; an immediate rerun from the same
short path completed the full tests and APK build successfully.
