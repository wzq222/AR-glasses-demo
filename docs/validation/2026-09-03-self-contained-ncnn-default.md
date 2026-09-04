# Self-contained NCNN default build

## Outcome

The Android project now builds the validated mobile proposal chain directly after a
normal Git LFS checkout. The default backend is the 512 single-class YOLOv8s-P2 NCNN
FP32 detector followed by the 128 MobileNetV3-Small marked-point verifier. The
experimental witness ROI model remains the third, assistive stage and still requires
operator confirmation.

No `CRRC_NCNN_ANDROID_ROOT`, `CRRC_NCNN_MODEL_DIR`, or
`CRRC_NCNN_JNI_BUILD_ROOT` variables are required for the default build. Setting both
native build roots remains available for controlled JNI rebuild experiments. Setting
`CRRC_DETECTOR_BACKEND=onnx` selects the 640 ONNX fallback.

## Bundled runtime inputs

| File | Bytes | SHA-256 |
|---|---:|---|
| `ncnnAssets/model.ncnn.param` | 21,107 | `EE68160881FE607CCE87485E569095A917A1511394BE66F39FE7567EFE4C9BB0` |
| `ncnnAssets/model.ncnn.bin` | 42,768,268 | `ED1448C049809A4E8E2D1D2AFD254AAE66AA4C1238D70B1CA6D9C2835DE9DCEC` |
| `ncnnJniLibs/arm64-v8a/libcrrc_ncnn.so` | 38,303,536 | `BBE8628F00298FF9D8DDA85697696F048488EDD7BFFE8F46A9DD562ED11D0CDD` |
| `ncnnJniLibs/armeabi-v7a/libcrrc_ncnn.so` | 18,978,288 | `CC009AF1D2B945BD4ACA22E63E0062B9B56AE455FF7539E32913DAD2A1203C79` |
| `assets/marked-point-verifier.onnx` | 6,089,246 | `FED197A11134DD4358B70EFF64086C050DDECC9B2C484E72AAEB102E4BA563CD` |
| `assets/witness-roi.onnx` | 5,458,941 | `6D42E0D6C5785866DC65077FCD4D5E6EED576689431CA5C3E6649A280A5880BA` |

The legacy proposal model remains available at
`onnxAssets/fastener-target-p2-640.onnx`, but it is excluded from the default APK.
All large model and JNI artifacts use Git LFS. The pre-build gate checks exact size
and SHA-256 before compiling.

## Verification

- Clean default `testDebugUnitTest assembleDebug`: PASS.
- JVM tests: 55, failures 0, errors 0.
- Generated `BuildConfig`: `DETECTOR_BACKEND="ncnn"` and
  `MARKED_POINT_VERIFIER_ENABLED=true`.
- Default APK contains both NCNN model files, both ABI JNI libraries, verifier ONNX,
  and witness ONNX; it does not contain `fastener-target-p2-640.onnx`.
- Default APK: 108,787,005 bytes.
- Default APK SHA-256:
  `73D7614AE6FC7A9C8099CA69A54B8A1AF2C3C8007E072C031E8DADFD5D10AA43`.
- Explicit ONNX fallback clean build: PASS. Its APK contains the 640 ONNX detector
  and excludes the NCNN model and JNI libraries.

ADB reported no connected device, so this change has not yet repeated the P20 Pro
installation, frozen-image parity run, or hot 50-run latency measurement. The earlier
75/75 development coverage is not an independent production accuracy claim, and the
witness ROI state model remains an assistive, fail-closed model rather than an
automatic looseness classifier.
