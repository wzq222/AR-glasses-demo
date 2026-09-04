# Bundled Case Gallery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package seven Git-external case images into the debug APK and expose them through the existing App original-gallery picker.

**Architecture:** A pure Java seeder recursively copies supported files from an abstract APK asset source into an app-internal case directory. `GalleryActivity` merges that directory with the existing external original gallery. Gradle conditionally packages images from `CRRC_BUILTIN_GALLERY_DIR` into generated APK assets without tracking field images in Git.

**Tech Stack:** Android Java, `AssetManager`, Gradle 7 `Sync`, JUnit 4, ADB.

---

### Task 1: Safe idempotent case seeder

**Files:**
- Create: `app/src/main/java/com/ar/glass/ui/BuiltinGallerySeeder.java`
- Test: `app/src/test/java/com/ar/glass/ui/BuiltinGallerySeederTest.java`

- [ ] **Step 1: Write the failing recursive-copy test**

Create a fake `BuiltinGallerySeeder.Source` containing `qr/QR-REAL-01.jpg`,
`fastener/LOCK-REAL-01.jpg`, `meter/METER-04_zero_000.jpg` and a rejected
`meter/manifest.csv`. Assert that `seed` returns three, preserves subdirectories,
copies bytes, and a second run returns zero.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
.\gradlew.bat :app:testDebugUnitTest --tests com.ar.glass.ui.BuiltinGallerySeederTest
```

Expected: compilation failure because `BuiltinGallerySeeder` does not exist.

- [ ] **Step 3: Implement the minimum seeder**

Expose:

```java
public interface Source {
    String[] list(String relativePath) throws IOException;
    InputStream open(String relativePath) throws IOException;
}

public static int seed(Source source, File targetRoot) throws IOException
```

Recurse only through safe path segments, accept JPEG/PNG/WebP, write a sibling
`.part` file, skip an existing same-size target, and rename only after a complete copy.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the Task 1 command. Expected: all `BuiltinGallerySeederTest` methods pass.

- [ ] **Step 5: Commit the seeder slice**

```powershell
git add app/src/main/java/com/ar/glass/ui/BuiltinGallerySeeder.java app/src/test/java/com/ar/glass/ui/BuiltinGallerySeederTest.java
git commit -m "feat(android): seed bundled gallery cases safely"
```

### Task 2: APK build-time image packaging

**Files:**
- Modify: `app/build.gradle`
- Test: `app/src/test/java/com/ar/glass/ui/BundledGalleryBuildContractTest.java`

- [ ] **Step 1: Write the failing Gradle contract test**

Assert that `app/build.gradle` declares `CRRC_BUILTIN_GALLERY_DIR`, a generated
asset directory, a `prepareBuiltinGalleryAssets` task, image-only includes, and a
`preBuild` dependency.

- [ ] **Step 2: Run the focused test and verify RED**

```powershell
.\gradlew.bat :app:testDebugUnitTest --tests com.ar.glass.ui.BundledGalleryBuildContractTest
```

Expected: assertion failure because the packaging contract is absent.

- [ ] **Step 3: Add conditional generated assets**

Read the external directory only from `CRRC_BUILTIN_GALLERY_DIR`. Configure a `Sync`
task that copies only `**/*.jpg`, `**/*.jpeg`, `**/*.png`, and `**/*.webp` beneath
`builtin_gallery`, adds the generated directory to `sourceSets.main.assets`, and runs
before `preBuild`. Missing configuration produces an APK without built-in cases rather
than reading a hard-coded private path.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the Task 2 command. Expected: the build contract passes.

- [ ] **Step 5: Commit the packaging slice**

```powershell
git add app/build.gradle app/src/test/java/com/ar/glass/ui/BundledGalleryBuildContractTest.java
git commit -m "build(android): package optional gallery cases"
```

### Task 3: Merge built-in cases into the original gallery

**Files:**
- Modify: `app/src/main/java/com/ar/glass/ui/GalleryActivity.java`
- Modify: `app/src/test/java/com/ar/glass/ui/GallerySelectionContractTest.java`

- [ ] **Step 1: Extend the gallery contract test**

Require `GalleryActivity` to create `filesDir/builtin_gallery/v1`, adapt
`AssetManager` to `BuiltinGallerySeeder.Source`, seed off the main thread, and collect
both the built-in and external gallery roots before sorting.

- [ ] **Step 2: Run the focused test and verify RED**

```powershell
.\gradlew.bat :app:testDebugUnitTest --tests com.ar.glass.ui.GallerySelectionContractTest
```

Expected: assertion failure because the built-in directory is not referenced.

- [ ] **Step 3: Integrate seeding and merged display**

Keep the current external `glass_media/photos` directory. Add a separate internal
`builtin_gallery/v1` directory, run the seeder inside the existing executor, collect
both roots, and post a short non-blocking message only if bundled extraction fails.

- [ ] **Step 4: Run focused and full tests**

```powershell
.\gradlew.bat :app:testDebugUnitTest --tests com.ar.glass.ui.*
.\gradlew.bat :app:testDebugUnitTest
```

Expected: all Android JVM tests pass.

- [ ] **Step 5: Commit the gallery slice**

```powershell
git add app/src/main/java/com/ar/glass/ui/GalleryActivity.java app/src/test/java/com/ar/glass/ui/GallerySelectionContractTest.java
git commit -m "feat(android): show bundled cases in original gallery"
```

### Task 4: Build, inspect, and install the APK

**Files:**
- Modify: `PROJECT_STATUS.md`
- Create: `docs/validation/2026-09-04-bundled-case-gallery-apk.md`

- [ ] **Step 1: Build with the Git-external image directory**

```powershell
$env:CRRC_BUILTIN_GALLERY_DIR='E:\Work\京新数智\识动hicool\中车眼镜数据资产\app-original-gallery'
.\gradlew.bat clean :app:assembleDebug
```

Expected: `BUILD SUCCESSFUL`.

- [ ] **Step 2: Inspect the APK contents**

Open the APK as ZIP and assert exactly seven files below `assets/builtin_gallery`:
one QR, one fastener, and five meter images. Record APK size and SHA-256.

- [ ] **Step 3: Install and smoke-test when a device is available**

```powershell
E:\Android\Sdk\platform-tools\adb.exe install -r app\build\outputs\apk\debug\app-debug.apk
```

Expected: `Success`; launch the App, open the original gallery, and verify the seven
case images are present. If no device is listed, record installation as blocked and
still return the validated APK.

- [ ] **Step 4: Record immutable evidence**

Write the exact seven filenames, APK hash, test/build results, device result, unchanged
model boundary, and unchanged formal-truth hash to the validation document and project
status.

- [ ] **Step 5: Commit and push**

```powershell
git add PROJECT_STATUS.md docs/validation/2026-09-04-bundled-case-gallery-apk.md
git commit -m "docs(android): validate bundled case gallery APK"
git push origin feature/sop-three-ten-review
```
