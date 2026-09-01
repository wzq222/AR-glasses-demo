# Android ncnn Benchmark Phase 2 Plan

**Goal:** Put the parity-passed ncnn FP32 model behind the same camera and postprocess path as
the current ONNX detector, then measure both implementations on the connected P20 Pro.

**Architecture:** Java owns the frozen letterbox and post-NMS contract. A small JNI library
loads ncnn `param/bin` assets, consumes reusable direct NCHW buffers, and writes the frozen
`6x34000` output into a reusable direct buffer. Build-time environment variables select the
benchmark backend and Git-external ncnn package/model directories; a normal checkout still
builds the ONNX default without private artifacts.

### Task 1: Introduce a shared detector lifecycle

- Add a small `FastenerDetector` interface without changing visible behavior.
- Make `OnnxFastenerDetector` implement it.
- Add a fail-closed backend selector and unit tests before wiring the activity.

### Task 2: Build pinned ncnn Android packages outside Git

- Build/install commit `2130e00c6efd910d3e926867ca94a2d96eaf9d31` for `arm64-v8a` and
  `armeabi-v7a` with NDK 26.3, CPU first and Vulkan disabled.
- Record toolchain, ABI, library, header, and model hashes.

### Task 3: Add the JNI detector

- Add failing Java contract tests for asset names, frozen shapes, lifecycle, and reusable
  direct buffers.
- Implement native create/infer/destroy with `AAssetManager`, four ncnn threads, `in0/out0`,
  shape checks, and path-free public errors.
- Reuse Java preprocessing and `YoloPostprocessor`; do not implement a second NMS policy.

### Task 4: Build and install the ncnn APK

- Inject the Git-external ncnn model and package only for the benchmark build.
- Run Java/Python tests and assemble the APK.
- Verify APK model/native hashes, install explicitly on `CLT-AL00`, and keep the glasses
  serial out of the benchmark command.

### Task 5: Run the phone A/B gate

- Use the same camera resolution, one-second cooldown, and overlay path.
- Capture cold, 50-result steady, and ten-minute hot latency plus PSS and thermal evidence.
- Require no initialization/inference failures and visually verify detections on the same
  fixed test scenes before considering speed.
- Keep ONNX as default unless ncnn preserves recall evidence and materially improves hot P95.

### Task 6: Try Vulkan only after CPU correctness

- Build a separate Vulkan-enabled ncnn package and APK.
- Repeat the exact parity and phone gates; never replace CPU measurements with desktop claims.
