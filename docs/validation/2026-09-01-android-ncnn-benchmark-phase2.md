# Android ncnn Benchmark Phase 2

## Decision

The FP32 ncnn CPU backend passes the same-device runtime parity gate against ONNX Runtime on the
Huawei P20 Pro (`CLT-AL00`, serial `TPC7N18604005991`). It remains the installed mobile performance
baseline. Both Vulkan variants preserve the 17-image detection-count gate, but neither provides a
material speed win: full FP32 is much slower, while full FP16 improves total P50 by only about 3%.
None of these variants is real-time or changes the detector's production-accuracy status.

The low-threshold calibration and exact-math variants are rejected:

- lowering confidence from `0.20` to `0.15` recovers one desktop-edge candidate but changes the
  17-image result from 81 to 94 detections, adding too many low-score candidates;
- disabling packing, Winograd, SGEMM, and denormal flushing leaves same-device output unchanged but
  slows inference to roughly 2.1–2.4 seconds.

## Frozen inputs

- Model: YOLOv8s-P2 FP32, `1x3x640x640 -> 1x6x34000`
- Source ONNX SHA-256:
  `C50F9105FF75885BE3BA02464E6A994FA7A45FDE0B0634AEA12FAA04A6CC5B7A`
- ncnn FP32 bin SHA-256:
  `0BBE90B8A2D916DEA1F42DB46A4035A2A024242D059269C6C22283BE7F3BF3F9`
- ncnn param SHA-256:
  `73F3A45150D559FECA8287D7EBCE649DAC385423CA826FDFA0216E287F375BFB`
- Validation: the same 17 complete `marked-point-v1.4` images, decoded and letterboxed by the same
  Android implementation for both runtimes
- Confidence/NMS: `0.20 / 0.45`, class-agnostic NMS, top 1000, maximum 100 detections
- Device: Huawei P20 Pro, Android CPU, four inference threads

## Same-device parity

The Android ONNX and fast ncnn variants both produce 81 post-NMS detections with identical
per-image counts:

`11, 4, 5, 10, 2, 9, 3, 0, 2, 2, 1, 2, 7, 1, 4, 8, 10`.

Logged coordinates agree to the displayed `0.001 px` precision and scores agree to the displayed
six decimal places apart from floating-point tail differences. Therefore ncnn introduces no
additional missing or unexpected detection relative to the existing Android ONNX path.

The desktop adapter reports 82 detections because one image-418 candidate scores `0.201771` there;
the common Android JPEG decode/resize path moves that edge candidate below `0.20` for both phone
runtimes. This is a mobile-input robustness issue, not an ncnn conversion loss. It is not waived as
evidence of production recall, and threshold reduction is rejected by the false-positive result
above.

## P20 Pro timing

| Variant | Observed result | Decision |
|---|---:|---|
| ONNX Runtime FP32, 17-image run | approximately 1.65 s steady per image | baseline |
| ncnn FP32 fast, 17-image run | P50 1049.8 ms; range 965.1–1071.2 ms | parity pass, performance baseline |
| ncnn FP32 exact-math | approximately 2.1–2.4 s per image | rejected |
| ncnn Vulkan FP32, 17-image run | steady total about 3.0 s; inference about 2.9 s | parity-count pass, performance rejected |
| ncnn Vulkan FP16, 17-image run | steady P50/P95 1019.9/1041.2 ms; inference P50 921.9 ms | parity-count pass, gain too small |
| ncnn FP32 CPU fast after Vulkan-option integration | P50/P95 1051.9/1082.8 ms; inference P50 973.3 ms | 81 detections, restored and installed |

The CPU fast path reduces steady full-image latency by roughly one third versus same-device ONNX,
but at about one inference per second it does not satisfy a real-time camera goal. The P20 Pro
advertises Vulkan compute and ncnn identifies its Mali-G72 correctly. Full FP32 Vulkan nevertheless
slows inference to roughly 2.9 seconds. Enabling FP16 packing, storage, and arithmetic restores the
17 frozen per-image counts and reaches 81 total detections, but its steady total P50 is only 29.9 ms
better than CPU. This is not a material win, so the CPU APK was restored after the experiment.
Trained single-class lightweight challengers remain required.

## Android implementation

- `FastenerDetector` provides one lifecycle/detection contract for ONNX and ncnn.
- `FastenerInputWorkspace` is reused by both backends, so letterbox and NCHW conversion are shared.
- ncnn uses reusable direct input/output buffers and JNI; Java retains the frozen postprocessor.
- arm64-v8a and armeabi-v7a ncnn/JNI libraries are built outside Git with NDK `25.1.8937393`.
- Build-time variables select the backend and Git-external model/runtime roots; default source builds
  remain ONNX-compatible and do not package Git-external artifacts.
- `CRRC_NCNN_VULKAN` and `CRRC_NCNN_VULKAN_FP16` make the two GPU experiments explicit and
  reproducible without changing the default CPU behavior.
- A debug-only offline benchmark activity can run internal frozen images without camera timing noise.

## Artifacts and invariants

- Git-external Android build root: `tools/crrc-ncnn-jni/`
- Git-external staged parity images: `runs/mobile-android-parity-v1/images/`
- Git-external APKs: `runs/mobile-android-benchmark-v1/apk/`
- ONNX debug APK SHA-256:
  `45EC5E3FA268306F95BE1BCA07F0F9ABC9F197A33A872842C52C46892A00F480`
- Original ncnn fast debug APK SHA-256:
  `49A5D4ED95D6857987F24293868CC4757CB1ED16084D70DC5683B8665A23DEDD`
- Final installed CPU APK with reproducible Vulkan options SHA-256:
  `C9457D8CB72DC90A8F84F91C671C1D1F667F8EFBC2164459DC3D34DCBE90F76C`
- ncnn Vulkan FP32 debug APK SHA-256:
  `E8D4AA89E248E4C8F0F6698C5235756E16F25C2F074E13088C3E66E76BA3E6FA`
- ncnn Vulkan FP16 debug APK SHA-256:
  `9ADF701D92D11DA410643CE53A516443F60C9764A7596998FC1D19D8F377F098`
- Android unit tests and default debug APK build: passed
- Formal truth SHA-256 after the run:
  `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`

This benchmark validates runtime replacement and speed only. It does not make the current detector
production-accurate and does not implement reliable anti-loosening state classification.
