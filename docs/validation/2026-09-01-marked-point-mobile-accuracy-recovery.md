# Marked-point mobile accuracy and latency recovery

## Decision

The phone candidate path now uses a dedicated `marked_point` YOLOv8s-P2 detector at
`512x512`, followed by a scene-isolated, three-seed MobileNetV3-Small verifier model soup at
`128x128`. The detector runs in ncnn FP32. The verifier runs one ROI per ONNX Runtime call
with the XNNPACK execution provider; the single-ROI route is required because this P20 Pro's
XNNPACK path returned an invalid batch for a dynamic multi-ROI input.

This is the highest-recall mobile route validated in this phase. It detects inspection points
that carry an anti-loosening mark. It does **not** determine whether a fastener is loose. A
production `ALIGNED/DISPLACED` claim still requires controlled real before/after pairs and a
state-model quality gate.

## Frozen evidence

- Formal truth SHA-256 before and after every run:
  `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`
- Development validation: 17 complete scenes, 75 marked-point truth objects, scene isolated
  from training.
- Detector ONNX SHA-256:
  `EE1BA5B432FEC8BDE74114A9203B778E0CE6365A43D52778C0AAEFBDE09DB79C`
- Detector ncnn param/bin SHA-256:
  `EE68160881FE607CCE87485E569095A917A1511394BE66F39FE7567EFE4C9BB0` /
  `ED1448C049809A4E8E2D1D2AFD254AAE66AA4C1238D70B1CA6D9C2835DE9DCEC`
- Verifier three-seed soup checkpoint SHA-256:
  `106AC9A43576DFFEDFA1F52793E01CE8B74DB6F53CA45B8804BF6F9738C4244A`
- Dynamic verifier ONNX SHA-256:
  `FED197A11134DD4358B70EFF64086C050DDECC9B2C484E72AAEB102E4BA563CD`
- Final debug APK SHA-256:
  `39AAAD13B09264E1F267003D666B001588F7C7FC8D0E0D79046A17907A45BC99`
- Phone under test: Huawei P20 Pro `CLT-AL00`, serial `TPC7N18604005991`. No APK was
  installed on the glasses serial.

## Desktop accuracy and runtime parity

At detector confidence `0.0019424824276939034`, NMS IoU `0.70`, and maximum 300 boxes:

- 512 ONNX and ncnn both produced 1,242 proposals.
- The proposal gate covered 75/75 truth objects.
- Detector ONNX/ncnn parity had 0 missing and 0 unexpected boxes; maximum coordinate drift
  was 0.001252 px and maximum score drift was 0.00000197.
- The 128 verifier soup used verifier threshold `0.2819833755493164`, proposal bypass
  `0.9676928520212636`, and semantic NMS IoU `0.30`.
- It retained 287 proposals, 16.88 per image, while covering 75/75 truth objects.
- Verifier ONNX/ncnn parity over all 1,242 real proposal crops had 0 threshold flips, maximum
  score drift 0.0001815, identical 287-box selection, and identical 75/75 coverage.

The 128 model is a three-seed equal-weight state-dict soup. Individual seeds were unstable:
only one of three passed the combined recall/burden gate. The soup is used to reduce that
single-seed variance; the sealed test remains unopened.

## P20 Pro result

The final XNNPACK build was evaluated by parsing every detailed box emitted by the installed
APK, not by copying desktop predictions. The final run used unique token
`final-20260901-xnnpack-r2`; the gate required exactly one summary per expected image, box
counts and indices matching each summary, and exactly one 17-image completion marker. Its
run-contract error list was empty, so retained logcat data could not complete or mix this run:

- 17/17 complete scenes observed.
- 293 final phone detections.
- 75/75 truth objects covered; development-set proposal recall `1.000`.
- Final-APK end-to-end P50/P95 `1.370/2.541 s` per 2000x1500 image with the display awake.
- Final-APK inference P50/P95 `1.082/1.782 s`.

These timings are inspection-mode latency, not camera-frame-rate realtime. The app remains
usable for human-confirmed capture-and-review, but the current P20 Pro cannot claim smooth
video inference.

## Rejected challengers

- Lowering detector confidence to `0.001` increased raw proposals from 1,242 to 2,671. The
  fixed recall-safe verifier setting kept 75/75 only at about 25.1 candidates/image, so the
  extra ARM score margin was rejected as excessive burden.
- A newly trained 96-input verifier reached only 74/75 after semantic NMS. Recovering 75/75
  required NMS IoU `0.60` and 26.29 candidates/image; it was rejected.
- NNAPI produced an unrecognized-device message on this phone and P50 `1.752 s`, effectively
  the same as the CPU verifier P50 of about `1.756 s`; it was rejected.
- XNNPACK dynamic batches returned an incompatible output batch. Single-ROI XNNPACK kept the
  same 293 detections and 75/75 phone coverage while reducing final-run P50 to `1.370 s`, so it is
  the selected verifier runtime.

## Verification

- Python: `412 passed`.
- Android JVM: `72 tests`, 0 failures, 0 errors.
- Clean Android debug build: PASS.
- Runtime exceptions clear the visible overlay and latch inference until the activity is
  explicitly restarted, preventing stale boxes from being shown over newer camera frames.
- Tokenized Android gate artifact:
  `E:\crrc_vision_data\runs\android-marked-point-512-verifier128-xnnpack-tokenized-final\coverage-and-timing.json`.
- APK-contained detector param/bin and verifier ONNX hashes exactly match the frozen artifacts
  listed above.
- Formal truth remained unchanged.

## Remaining accuracy boundary

The 75/75 result is a recall result on a small same-source development validation split. It
does not establish cross-vehicle precision or production accuracy. Before production use,
freeze the model and thresholds, collect at least 100-150 complete scenes across vehicles,
phones, lighting, blur, distance, and occlusion, and open a one-time independent sealed test.
For looseness state, collect controlled aligned/displaced physical pairs by inspection point;
synthetic images and unpaired historical images remain training aids only.
