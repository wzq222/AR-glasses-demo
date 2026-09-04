# SOP Gallery Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add phone-gallery and app-original-gallery evidence sources to every SOP step, then install and validate a complete three-step run on the connected phone.

**Architecture:** A pure Java evidence-file contract validates supported image types and creates immutable run-specific snapshots. `SopActivity` adapts camera, system-picker and app-gallery results into that shared path, while `GalleryActivity` gains an explicit result-returning selection mode without changing normal browsing.

**Tech Stack:** Android Java, Activity Result APIs, AndroidX ExifInterface, JUnit 4, Gradle, ADB/UIAutomator, FastAPI production backend.

---

### Task 1: Evidence snapshot contract

**Files:**
- Create: `app/src/main/java/com/ar/glass/sop/SopEvidenceFiles.java`
- Test: `app/src/test/java/com/ar/glass/sop/SopEvidenceFilesTest.java`

- [ ] **Step 1: Write failing tests for validated image snapshots**

```java
@Test public void copiesSupportedImageIntoUniqueEvidenceFile() throws Exception {
    File file = SopEvidenceFiles.copy(
            new ByteArrayInputStream(new byte[]{1, 2, 3}),
            "image/png", temporaryFolder.getRoot(), 1234L);
    assertEquals("sop_1234.png", file.getName());
    assertArrayEquals(new byte[]{1, 2, 3}, Files.readAllBytes(file.toPath()));
}

@Test public void rejectsNonImageAndMoreThanTwentyFiveMib() throws Exception {
    assertThrows(IllegalArgumentException.class, () ->
            SopEvidenceFiles.extensionFor("video/mp4"));
    assertThrows(IOException.class, () -> SopEvidenceFiles.copy(
            new RepeatingInputStream(25 * 1024 * 1024 + 1),
            "image/jpeg", temporaryFolder.getRoot(), 1L));
}
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `./gradlew.bat :app:testDebugUnitTest --tests com.ar.glass.sop.SopEvidenceFilesTest`

Expected: compilation failure because `SopEvidenceFiles` does not exist.

- [ ] **Step 3: Implement the smallest snapshot utility**

```java
public final class SopEvidenceFiles {
    public static final long MAX_BYTES = 25L * 1024L * 1024L;

    public static String extensionFor(String mime) {
        if ("image/jpeg".equals(mime)) return ".jpg";
        if ("image/png".equals(mime)) return ".png";
        if ("image/webp".equals(mime)) return ".webp";
        throw new IllegalArgumentException("仅支持 JPEG、PNG 或 WebP 图片");
    }

    public static File copy(InputStream input, String mime, File directory, long now)
            throws IOException {
        if (input == null) throw new IOException("无法读取所选图片");
        if (!directory.exists() && !directory.mkdirs()) {
            throw new IOException("无法创建证据目录");
        }
        File output = unique(directory, "sop_" + now, extensionFor(mime));
        long total = 0;
        try (InputStream source = input; FileOutputStream target = new FileOutputStream(output)) {
            byte[] buffer = new byte[8192];
            int count;
            while ((count = source.read(buffer)) != -1) {
                total += count;
                if (total > MAX_BYTES) throw new IOException("图片超过25 MiB");
                target.write(buffer, 0, count);
            }
        } catch (Exception error) {
            output.delete();
            if (error instanceof IOException) throw (IOException) error;
            throw new IOException(error.getMessage(), error);
        }
        if (total == 0) {
            output.delete();
            throw new IOException("所选图片为空");
        }
        return output;
    }

    public static String mediaType(File file) {
        String name = file.getName().toLowerCase(Locale.ROOT);
        if (name.endsWith(".jpg") || name.endsWith(".jpeg")) return "image/jpeg";
        if (name.endsWith(".png")) return "image/png";
        if (name.endsWith(".webp")) return "image/webp";
        throw new IllegalArgumentException("不支持的证据图片类型");
    }

    private static File unique(File directory, String base, String extension) {
        File candidate = new File(directory, base + extension);
        for (int suffix = 1; candidate.exists(); suffix++) {
            candidate = new File(directory, base + "_" + suffix + extension);
        }
        return candidate;
    }
}
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Expected: all `SopEvidenceFilesTest` cases pass.

- [ ] **Step 5: Commit**

```powershell
git add app/src/main/java/com/ar/glass/sop/SopEvidenceFiles.java app/src/test/java/com/ar/glass/sop/SopEvidenceFilesTest.java
git commit -m "feat(android): add SOP evidence snapshots"
```

### Task 2: Bounded, orientation-correct image decoding

**Files:**
- Modify: `app/build.gradle`
- Create: `app/src/main/java/com/ar/glass/sop/SopImageDecoder.java`
- Test: `app/src/test/java/com/ar/glass/sop/SopImageDecoderContractTest.java`

- [ ] **Step 1: Write a failing source contract**

```java
@Test public void decoderBoundsAndAppliesExifOrientation() throws Exception {
    String source = read("app/src/main/java/com/ar/glass/sop/SopImageDecoder.java");
    assertTrue(source.contains("maxEdge"));
    assertTrue(source.contains("ExifInterface.TAG_ORIENTATION"));
    assertTrue(source.contains("Bitmap.createBitmap"));
}
```

- [ ] **Step 2: Run the focused test and verify RED**

Expected: failure because `SopImageDecoder.java` is absent.

- [ ] **Step 3: Add AndroidX ExifInterface and implement decoding**

```groovy
implementation 'androidx.exifinterface:exifinterface:1.3.7'
```

`SopImageDecoder.decode(File file, int maxEdge)` first probes dimensions, selects a
power-of-two sample, decodes ARGB_8888, reads EXIF orientation, then rotates or mirrors
the bitmap with a `Matrix`. It throws a descriptive `IOException` for invalid images.

```java
public static Bitmap decode(File file, int maxEdge) throws IOException {
    BitmapFactory.Options bounds = new BitmapFactory.Options();
    bounds.inJustDecodeBounds = true;
    BitmapFactory.decodeFile(file.getAbsolutePath(), bounds);
    if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
        throw new IOException("无法解析所选图片");
    }
    int sample = 1;
    while (Math.max(bounds.outWidth, bounds.outHeight) / sample > maxEdge) sample *= 2;
    BitmapFactory.Options options = new BitmapFactory.Options();
    options.inSampleSize = sample;
    options.inPreferredConfig = Bitmap.Config.ARGB_8888;
    Bitmap bitmap = BitmapFactory.decodeFile(file.getAbsolutePath(), options);
    if (bitmap == null) throw new IOException("无法解析所选图片");

    int orientation = new ExifInterface(file.getAbsolutePath()).getAttributeInt(
            ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
    Matrix matrix = matrixFor(orientation);
    if (matrix.isIdentity()) return bitmap;
    Bitmap oriented = Bitmap.createBitmap(
            bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    if (oriented != bitmap) bitmap.recycle();
    return oriented;
}

private static Matrix matrixFor(int orientation) {
    Matrix matrix = new Matrix();
    switch (orientation) {
        case ExifInterface.ORIENTATION_FLIP_HORIZONTAL: matrix.setScale(-1f, 1f); break;
        case ExifInterface.ORIENTATION_ROTATE_180: matrix.setRotate(180f); break;
        case ExifInterface.ORIENTATION_FLIP_VERTICAL: matrix.setScale(1f, -1f); break;
        case ExifInterface.ORIENTATION_TRANSPOSE:
            matrix.setRotate(90f); matrix.postScale(-1f, 1f); break;
        case ExifInterface.ORIENTATION_ROTATE_90: matrix.setRotate(90f); break;
        case ExifInterface.ORIENTATION_TRANSVERSE:
            matrix.setRotate(-90f); matrix.postScale(-1f, 1f); break;
        case ExifInterface.ORIENTATION_ROTATE_270: matrix.setRotate(-90f); break;
        default: break;
    }
    return matrix;
}
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Expected: the decoder contract passes and `:app:compileDebugJavaWithJavac` succeeds.

- [ ] **Step 5: Commit**

```powershell
git add app/build.gradle app/src/main/java/com/ar/glass/sop/SopImageDecoder.java app/src/test/java/com/ar/glass/sop/SopImageDecoderContractTest.java
git commit -m "feat(android): decode SOP evidence safely"
```

### Task 3: App original-gallery selection mode

**Files:**
- Modify: `app/src/main/java/com/ar/glass/ui/GalleryActivity.java`
- Test: `app/src/test/java/com/ar/glass/ui/GallerySelectionContractTest.java`

- [ ] **Step 1: Write failing tests for browse and selection modes**

```java
@Test public void selectionModeReturnsChosenPath() throws Exception {
    String source = read("app/src/main/java/com/ar/glass/ui/GalleryActivity.java");
    assertTrue(source.contains("EXTRA_SELECT_IMAGE"));
    assertTrue(source.contains("EXTRA_SELECTED_IMAGE_PATH"));
    assertTrue(source.contains("setResult(RESULT_OK"));
    assertTrue(source.contains("returnSelectedImage"));
}
```

- [ ] **Step 2: Run the focused test and verify RED**

Expected: assertion failure because selection-mode constants and return path are absent.

- [ ] **Step 3: Implement explicit selection mode**

```java
public static final String EXTRA_SELECT_IMAGE = "select_image";
public static final String EXTRA_SELECTED_IMAGE_PATH = "selected_image_path";

private void openOrReturnImage(int index) {
    if (getIntent().getBooleanExtra(EXTRA_SELECT_IMAGE, false)) {
        Intent result = new Intent().putExtra(
                EXTRA_SELECTED_IMAGE_PATH, imageFiles.get(index).getAbsolutePath());
        setResult(RESULT_OK, result);
        finish();
        return;
    }
    openImageViewer(index);
}
```

Selection mode changes the title to `选择原图库图片` but retains `导入`, allowing an
empty original library to be populated from the system gallery.

- [ ] **Step 4: Run the focused test and verify GREEN**

Expected: `GallerySelectionContractTest` passes.

- [ ] **Step 5: Commit**

```powershell
git add app/src/main/java/com/ar/glass/ui/GalleryActivity.java app/src/test/java/com/ar/glass/ui/GallerySelectionContractTest.java
git commit -m "feat(android): select SOP evidence from original gallery"
```

### Task 4: Wire all three evidence sources into SOP execution

**Files:**
- Modify: `app/src/main/res/layout/activity_sop.xml`
- Modify: `app/src/main/java/com/ar/glass/sop/SopActivity.java`
- Modify: `app/src/main/java/com/ar/glass/sop/SopApiClient.java`
- Test: `app/src/test/java/com/ar/glass/sop/SopGalleryEvidenceContractTest.java`

- [ ] **Step 1: Write the failing UI and routing contract**

```java
@Test public void sopOffersBothGallerySourcesAndSharedAnalysis() throws Exception {
    String layout = read("app/src/main/res/layout/activity_sop.xml");
    String activity = read("app/src/main/java/com/ar/glass/sop/SopActivity.java");
    assertTrue(layout.contains("btnSelectPhoneGallery"));
    assertTrue(layout.contains("btnSelectOriginalGallery"));
    assertTrue(activity.contains("ActivityResultContracts.GetContent"));
    assertTrue(activity.contains("GalleryActivity.EXTRA_SELECT_IMAGE"));
    assertTrue(activity.contains("processEvidenceFile"));
    assertTrue(activity.contains("evidence_source"));
}
```

- [ ] **Step 2: Run the focused test and verify RED**

Expected: assertions fail because the buttons and shared routing method are absent.

- [ ] **Step 3: Add the two secondary buttons**

Add a horizontal row below `btnCaptureStep` with equally weighted
`btnSelectPhoneGallery` and `btnSelectOriginalGallery`. Text is final product copy:
`从手机图库选择` and `从原图库选择`.

```xml
<LinearLayout
    android:id="@+id/layoutGalleryEvidenceSources"
    android:layout_width="match_parent"
    android:layout_height="wrap_content"
    android:orientation="horizontal">
    <Button
        android:id="@+id/btnSelectPhoneGallery"
        android:layout_width="0dp"
        android:layout_height="wrap_content"
        android:layout_weight="1"
        android:text="从手机图库选择" />
    <Button
        android:id="@+id/btnSelectOriginalGallery"
        android:layout_width="0dp"
        android:layout_height="wrap_content"
        android:layout_weight="1"
        android:text="从原图库选择" />
</LinearLayout>
```

- [ ] **Step 4: Register and handle both picker results**

```java
phoneGallery = registerForActivityResult(new ActivityResultContracts.GetContent(),
        uri -> importPhoneEvidence(uri));
originalGallery = registerForActivityResult(
        new ActivityResultContracts.StartActivityForResult(),
        result -> importOriginalEvidence(result));
```

Both handlers snapshot the source through `SopEvidenceFiles`, decode through
`SopImageDecoder`, then call `processEvidenceFile(file, bitmap, source)`. Successful
replacement clears overlays and witness decisions; cancellation does nothing.

```java
private void processEvidenceFile(File file, Bitmap bitmap, String source) {
    evidenceFile = file;
    evidenceSource = source;
    evidenceOverlay.clear();
    resetWitnessReviews();
    saveStepButton.setEnabled(false);
    evidenceBitmap = bitmap;
    evidencePreview.setImageBitmap(bitmap);
    analyzeCurrentStep(bitmap);
}

private void importPhoneEvidence(Uri uri) {
    if (uri == null) return;
    importEvidence(() -> getContentResolver().openInputStream(uri),
            getContentResolver().getType(uri), "PHONE_GALLERY");
}

private void openOriginalGallery() {
    Intent intent = new Intent(this, GalleryActivity.class)
            .putExtra(GalleryActivity.EXTRA_MODE, GalleryActivity.MODE_ORIGINAL)
            .putExtra(GalleryActivity.EXTRA_SELECT_IMAGE, true);
    originalGallery.launch(intent);
}
```

- [ ] **Step 5: Preserve source metadata and correct upload media type**

`processEvidenceFile` stores `CAMERA`, `PHONE_GALLERY` or `APP_GALLERY`; analyzer
payload creation adds `evidence_source`. `SopApiClient.uploadEvidence` calls
`SopEvidenceFiles.mediaType(file)` instead of always sending `image/jpeg`.

```java
value.put("evidence_source", evidenceSource);
MediaType mediaType = MediaType.parse(SopEvidenceFiles.mediaType(file));
RequestBody body = new MultipartBody.Builder().setType(MultipartBody.FORM)
        .addFormDataPart("file", file.getName(), RequestBody.create(mediaType, file))
        .build();
```

- [ ] **Step 6: Run focused and full unit tests**

Run:

```powershell
./gradlew.bat :app:testDebugUnitTest --tests com.ar.glass.sop.SopGalleryEvidenceContractTest
./gradlew.bat :app:testDebugUnitTest
```

Expected: all tests pass with zero failures.

- [ ] **Step 7: Commit**

```powershell
git add app/src/main/res/layout/activity_sop.xml app/src/main/java/com/ar/glass/sop/SopActivity.java app/src/main/java/com/ar/glass/sop/SopApiClient.java app/src/test/java/com/ar/glass/sop/SopGalleryEvidenceContractTest.java
git commit -m "feat(android): select gallery evidence in SOP steps"
```

### Task 5: Build, install and run the production three-step flow

**Files:**
- Modify: `docs/validation/2026-09-04-three-ten-step-sop-and-inline-review.md`
- Modify: `PROJECT_STATUS.md`

- [ ] **Step 1: Run complete verification and build**

```powershell
./gradlew.bat :app:testDebugUnitTest :app:assembleDebug --rerun-tasks
python -m pytest tests -q  # from server/
git diff --check
```

Expected: Android and server tests pass; debug APK builds; diff check is clean.

- [ ] **Step 2: Install on the connected physical phone**

```powershell
E:/Android/Sdk/platform-tools/adb.exe devices -l
E:/Android/Sdk/platform-tools/adb.exe install -r -g app/build/outputs/apk/debug/app-debug.apk
```

Expected: `NOH-AN01` is online and install returns `Success`.

- [ ] **Step 3: Validate both picker entries**

Use UIAutomator to confirm both final button labels, open the phone picker, cancel
without losing state, open the app original gallery, and import at least one existing
phone image if its baseline count is zero. Do not copy any test image into Git.

- [ ] **Step 4: Complete `CRRC-DEMO-001`**

Use existing phone/original-gallery images for QR, witness and meter steps. Confirm
each analyzer runs, choose `无法判断` where the selected test image lacks valid
evidence, complete every witness-point review if candidates exist, save all three
steps, upload all evidence and submit the run.

- [ ] **Step 5: Verify the backend record**

Log into the public management backend in a disposable browser session. Confirm the
run is completed, all three steps and evidence objects exist, and source metadata is
present. Log out and close the browser; never persist or print credentials/tokens.

- [ ] **Step 6: Record hashes and verified limits, then commit and push**

Document APK size/SHA-256, test counts, phone model, step/evidence counts, backend
record state and any real inference limitations. Verify formal-truth SHA-256 remains
`B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`.

```powershell
git add PROJECT_STATUS.md docs/validation/2026-09-04-three-ten-step-sop-and-inline-review.md
git commit -m "docs(sop): validate gallery evidence on phone"
git push origin feature/sop-three-ten-review
```
