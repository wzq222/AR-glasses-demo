# Android Live Fastener Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and install a safe near-real-time fastener-candidate test APK on the connected Huawei P20 Pro.

**Architecture:** CameraX owns preview and latest-frame acquisition. A focused ONNX detector owns preprocessing and postprocessing, while a custom overlay only renders immutable detections. State text remains fail-closed because the deployed model has no state head.

**Tech Stack:** Android Java, CameraX 1.2.x, ONNX Runtime Android, YOLOv8s-P2 ONNX, JUnit 4, ADB.

---

### Task 1: Pure Java detection contract and postprocessing

**Files:**
- Create: `app/src/main/java/com/ar/glass/vision/realtime/Detection.java`
- Create: `app/src/main/java/com/ar/glass/vision/realtime/YoloPostprocessor.java`
- Test: `app/src/test/java/com/ar/glass/vision/realtime/YoloPostprocessorTest.java`

- [ ] Write failing tests for merged class score, confidence filtering, letterbox inverse mapping and class-agnostic NMS.
- [ ] Run `gradlew testDebugUnitTest --tests com.ar.glass.vision.realtime.YoloPostprocessorTest` and confirm failure.
- [ ] Implement immutable detection records and deterministic postprocessing for output shape `1x6x34000`.
- [ ] Re-run the targeted test and confirm pass.

### Task 2: ONNX model loading and bitmap inference

**Files:**
- Modify: `app/build.gradle`
- Create: `app/src/main/java/com/ar/glass/vision/realtime/OnnxFastenerDetector.java`
- Test: `app/src/test/java/com/ar/glass/vision/realtime/ModelAssetContractTest.java`

- [ ] Add CameraX and ONNX Runtime dependencies and optional `CRRC_VISION_MODEL_DIR` assets source.
- [ ] Write a failing contract test for the fixed asset name and tensor/output dimensions.
- [ ] Implement model loading, 640 letterbox, RGB CHW tensor creation, session execution and resource closing.
- [ ] Run targeted unit tests and an APK build without the model to verify graceful missing-model behavior.

### Task 3: Camera preview and overlay

**Files:**
- Create: `app/src/main/java/com/ar/glass/ui/LiveInspectionActivity.java`
- Create: `app/src/main/java/com/ar/glass/ui/DetectionOverlayView.java`
- Create: `app/src/main/res/layout/activity_live_inspection.xml`
- Modify: `app/src/main/AndroidManifest.xml`

- [ ] Implement PreviewView, overlay and fixed fail-closed state banner.
- [ ] Bind the rear camera with `KEEP_ONLY_LATEST`, one-in-flight inference and 500ms throttle.
- [ ] Convert RGBA analysis frames to correctly rotated bitmaps and map detections to preview coordinates.
- [ ] Release camera, executor and detector in lifecycle callbacks.

### Task 4: Main entry and product copy

**Files:**
- Modify: `app/src/main/res/layout/activity_main.xml`
- Modify: `app/src/main/java/com/ar/glass/ui/MainActivity.java`
- Modify: `app/src/main/res/values/strings.xml`

- [ ] Add a “手机实时检测” button that does not depend on BLE/eyeglass connection state.
- [ ] Add final user-facing copy for internal-model status, missing model, camera permission and safe refusal.
- [ ] Inspect all visible strings for misleading accuracy or loose/aligned claims.

### Task 5: Build, install and device validation

**Files:**
- Modify: `PROJECT_STATUS.md`
- Create: `docs/validation/2026-08-31-android-live-fastener-device.md`

- [ ] Run `python -m pytest ml/tests -q` and Android unit tests.
- [ ] Set `CRRC_VISION_MODEL_DIR=E:\crrc_vision_data\exports\android\yolov8s-p2-v3-640` and run a clean APK build.
- [ ] Verify the APK contains `fastener-target-p2-640.onnx` and record APK/model hashes.
- [ ] Install only with `adb -s TPC7N18604005991 install -r <apk>` and grant camera permission.
- [ ] Start the activity, capture logcat/screenshot, and record actual load time, inference latency and memory.
- [ ] Run full regression, inspect Git status, independently review the implementation, then commit.

