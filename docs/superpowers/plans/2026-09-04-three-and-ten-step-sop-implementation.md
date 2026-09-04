# Three- and Ten-Step SOP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver two production-visible SOP tasks and a mobile in-step witness-line reviewer that shows one centered 2.25× context crop with an explicit ROI box.

**Architecture:** Keep the 1.5× state-model crop unchanged and add a separate pure-geometry review crop. `SopActivity` renders one point at a time and persists decisions by point index. A repeatable deployment script publishes the ten-step template and creates only missing demo assignments.

**Tech Stack:** Android Java, Android `Bitmap`/`Canvas`, JUnit 4, FastAPI deployment API, Python `urllib`, pytest.

---

### Task 1: Pure review-crop geometry

**Files:**
- Create: `app/src/main/java/com/ar/glass/vision/ui/WitnessReviewCrop.java`
- Create: `app/src/test/java/com/ar/glass/vision/ui/WitnessReviewCropTest.java`

- [ ] **Step 1: Write the failing geometry tests**

```java
@Test public void usesTwoPointTwoFiveExpansionAndCentersRoi() {
    WitnessReviewCrop crop = WitnessReviewCrop.fromNormalized(
            0.40f, 0.30f, 0.45f, 0.36666667f, 2000, 1500);
    assertEquals(225f, crop.getSide(), 0.01f);
    assertEquals(0.5f, (crop.getRoiLeft() + crop.getRoiRight()) / 2f, 0.001f);
    assertEquals(0.5f, (crop.getRoiTop() + crop.getRoiBottom()) / 2f, 0.001f);
    assertFalse(crop.requiresCloserCapture());
}

@Test public void keepsTargetCenteredAtImageEdgeUsingPadding() {
    WitnessReviewCrop crop = WitnessReviewCrop.fromNormalized(
            0f, 0f, 0.04f, 0.04f, 2000, 1500);
    assertTrue(crop.getRequestedLeft() < 0f);
    assertTrue(crop.getRequestedTop() < 0f);
    assertEquals(0.5f, (crop.getRoiLeft() + crop.getRoiRight()) / 2f, 0.001f);
}

@Test public void flagsTargetsSmallerThanThirtyTwoSourcePixels() {
    WitnessReviewCrop crop = WitnessReviewCrop.fromNormalized(
            0.2f, 0.2f, 0.21f, 0.21f, 2000, 1500);
    assertTrue(crop.requiresCloserCapture());
}
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `./gradlew.bat :app:testDebugUnitTest --tests com.ar.glass.vision.ui.WitnessReviewCropTest`

Expected: compilation failure because `WitnessReviewCrop` does not exist.

- [ ] **Step 3: Implement the geometry value object**

```java
public final class WitnessReviewCrop {
    static final float EXPANSION = 2.25f;
    static final float MIN_REVIEWABLE_SIDE = 32f;
    private final float requestedLeft, requestedTop, side;
    private final float roiLeft, roiTop, roiRight, roiBottom;
    private final boolean requiresCloserCapture;

    private WitnessReviewCrop(
            float requestedLeft, float requestedTop, float side,
            float roiLeft, float roiTop, float roiRight, float roiBottom,
            boolean requiresCloserCapture) {
        this.requestedLeft = requestedLeft;
        this.requestedTop = requestedTop;
        this.side = side;
        this.roiLeft = roiLeft;
        this.roiTop = roiTop;
        this.roiRight = roiRight;
        this.roiBottom = roiBottom;
        this.requiresCloserCapture = requiresCloserCapture;
    }

    public static WitnessReviewCrop fromNormalized(
            float left, float top, float right, float bottom,
            int imageWidth, int imageHeight) {
        if (imageWidth <= 0 || imageHeight <= 0 || right <= left || bottom <= top) {
            throw new IllegalArgumentException("valid normalized ROI and image size required");
        }
        float x1 = left * imageWidth;
        float y1 = top * imageHeight;
        float x2 = right * imageWidth;
        float y2 = bottom * imageHeight;
        float targetSide = Math.max(x2 - x1, y2 - y1);
        float side = Math.max(1f, targetSide * EXPANSION);
        float cropLeft = (x1 + x2 - side) * 0.5f;
        float cropTop = (y1 + y2 - side) * 0.5f;
        return new WitnessReviewCrop(
                cropLeft, cropTop, side,
                (x1 - cropLeft) / side, (y1 - cropTop) / side,
                (x2 - cropLeft) / side, (y2 - cropTop) / side,
                targetSide < MIN_REVIEWABLE_SIDE);
    }
    public float getRequestedLeft() { return requestedLeft; }
    public float getRequestedTop() { return requestedTop; }
    public float getSide() { return side; }
    public float getRoiLeft() { return roiLeft; }
    public float getRoiTop() { return roiTop; }
    public float getRoiRight() { return roiRight; }
    public float getRoiBottom() { return roiBottom; }
    public boolean requiresCloserCapture() { return requiresCloserCapture; }
}
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `./gradlew.bat :app:testDebugUnitTest --tests com.ar.glass.vision.ui.WitnessReviewCropTest`

Expected: 3 tests, 0 failures.

- [ ] **Step 5: Commit Task 1**

```powershell
git add app/src/main/java/com/ar/glass/vision/ui/WitnessReviewCrop.java app/src/test/java/com/ar/glass/vision/ui/WitnessReviewCropTest.java
git commit -m "feat(android): add calibrated witness review crop"
```

### Task 2: One-point-at-a-time mobile review UI

**Files:**
- Modify: `app/src/main/res/layout/activity_sop.xml`
- Modify: `app/src/main/java/com/ar/glass/sop/SopActivity.java`
- Create: `app/src/main/java/com/ar/glass/vision/ui/WitnessReviewOverlay.java`
- Modify: `app/src/test/java/com/ar/glass/sop/SopWitnessReviewContractTest.java`

- [ ] **Step 1: Extend the contract test and verify RED**

```java
assertTrue(layout.contains("android:id=\"@+id/panelWitnessPointReview\""));
assertTrue(layout.contains("android:id=\"@+id/ivWitnessReviewCrop\""));
assertTrue(layout.contains("android:id=\"@+id/overlayWitnessReviewRoi\""));
assertTrue(layout.contains("android:id=\"@+id/btnWitnessPrevious\""));
assertTrue(layout.contains("android:id=\"@+id/btnWitnessNext\""));
assertTrue(layout.contains("android:id=\"@+id/btnWitnessFullImage\""));
assertTrue(source.contains("WitnessReviewCrop.fromNormalized"));
assertTrue(source.contains("renderWitnessReviewPoint"));
assertTrue(source.contains("showWitnessFullImage"));
```

Run: `./gradlew.bat :app:testDebugUnitTest --tests com.ar.glass.sop.SopWitnessReviewContractTest`

Expected: assertions fail because the review panel and navigation do not exist.

- [ ] **Step 2: Replace the dynamic list with a single review panel**

Add a hidden `LinearLayout` under `tvStepAnalysis` containing:

```xml
<TextView
    android:id="@+id/tvWitnessReviewProgress"
    android:layout_width="match_parent"
    android:layout_height="wrap_content"
    android:textColor="#243A48"
    android:textStyle="bold" />
<FrameLayout
    android:layout_width="240dp"
    android:layout_height="240dp"
    android:layout_gravity="center_horizontal">
    <ImageView
        android:id="@+id/ivWitnessReviewCrop"
        android:layout_width="match_parent"
        android:layout_height="match_parent"
        android:scaleType="fitCenter" />
    <com.ar.glass.vision.ui.WitnessReviewOverlay
        android:id="@+id/overlayWitnessReviewRoi"
        android:layout_width="match_parent"
        android:layout_height="match_parent" />
</FrameLayout>
<TextView
    android:id="@+id/tvWitnessAutomaticResult"
    android:layout_width="match_parent"
    android:layout_height="wrap_content"
    android:layout_marginTop="6dp"
    android:textColor="#20303A" />
<Spinner
    android:id="@+id/spinnerWitnessPointDecision"
    android:layout_width="match_parent"
    android:layout_height="wrap_content" />
<LinearLayout
    android:layout_width="match_parent"
    android:layout_height="wrap_content"
    android:orientation="horizontal">
    <Button
        android:id="@+id/btnWitnessPrevious"
        android:layout_width="0dp"
        android:layout_height="wrap_content"
        android:layout_weight="1"
        android:text="上一个" />
    <Button
        android:id="@+id/btnWitnessFullImage"
        android:layout_width="0dp"
        android:layout_height="wrap_content"
        android:layout_weight="1"
        android:text="查看原图" />
    <Button
        android:id="@+id/btnWitnessNext"
        android:layout_width="0dp"
        android:layout_height="wrap_content"
        android:layout_weight="1"
        android:text="下一个" />
</LinearLayout>
```

- [ ] **Step 3: Add crop rendering and indexed decision state**

Store the decoded evidence bitmap, the mapped detections, assessments, result maps, and an `int[]` selection array. `renderWitnessReviewPoint(index)` must:

```java
WitnessReviewCrop crop = WitnessReviewCrop.fromNormalized(
        assessment.left, assessment.top, assessment.right, assessment.bottom,
        evidenceBitmap.getWidth(), evidenceBitmap.getHeight());
Bitmap patch = renderSourceCropWithNeutralPadding(evidenceBitmap, crop);
witnessReviewImage.setImageBitmap(patch);
witnessReviewOverlay.setReview(
        crop, assessment.index, assessment.estimate.getTriage().name());
witnessReviewProgress.setText("检查点 " + (index + 1) + "/" + assessments.size());
witnessDecision.setSelection(pointSelections[index]);
```

`WitnessReviewOverlay` is a focused `View`: `setReview(WitnessReviewCrop crop, int index, String triage)` stores the crop-relative ROI coordinates, chooses the same triage color used by `BoxOverlay`, and draws a 3dp rectangle plus `检查点 N` label. It does not draw a filled mask over the fastener or witness line.

`renderSourceCropWithNeutralPadding` creates a square bitmap in source-pixel scale, fills it with neutral gray, copies only the intersecting source rectangle, and leaves padding on the clipped side so the ROI center stays at `(0.5, 0.5)`:

```java
private Bitmap renderSourceCropWithNeutralPadding(Bitmap source, WitnessReviewCrop crop) {
    int side = Math.max(1, Math.round(crop.getSide()));
    Bitmap output = Bitmap.createBitmap(side, side, Bitmap.Config.ARGB_8888);
    Canvas canvas = new Canvas(output);
    canvas.drawColor(Color.rgb(112, 112, 112));
    float left = crop.getRequestedLeft();
    float top = crop.getRequestedTop();
    Rect src = new Rect(
            Math.max(0, (int) Math.floor(left)),
            Math.max(0, (int) Math.floor(top)),
            Math.min(source.getWidth(), (int) Math.ceil(left + crop.getSide())),
            Math.min(source.getHeight(), (int) Math.ceil(top + crop.getSide())));
    RectF dst = new RectF(src.left - left, src.top - top,
            src.right - left, src.bottom - top);
    if (!src.isEmpty()) canvas.drawBitmap(source, src, dst, null);
    return output;
}
```

Recycle the previous patch only after the new patch has been attached to the `ImageView`.

- [ ] **Step 4: Add navigation, full-image view, and completion gate**

Previous/next buttons persist `spinnerWitnessPointDecision` into `pointSelections[currentIndex]` before navigation. “查看原图” opens a dialog with the original `evidenceBitmap` and a `BoxOverlay` containing only the current detection. Saving a `FASTENER_MARK` step loops over `pointSelections`; any zero value blocks saving with `请先确认每一个防松检查点`. Targets whose crop reports `requiresCloserCapture()` display `目标过小，请近拍` and default to “无法判断/重拍” without claiming an automatic state.

- [ ] **Step 5: Run tests and build**

Run: `./gradlew.bat :app:testDebugUnitTest assembleDebug`

Expected: all JVM tests pass and `app/build/outputs/apk/debug/app-debug.apk` exists.

- [ ] **Step 6: Commit Task 2**

```powershell
git add app/src/main/res/layout/activity_sop.xml app/src/main/java/com/ar/glass/sop/SopActivity.java app/src/main/java/com/ar/glass/vision/ui/WitnessReviewOverlay.java app/src/test/java/com/ar/glass/sop/SopWitnessReviewContractTest.java
git commit -m "feat(android): add centered witness point review"
```

### Task 3: Repeatable three- and ten-step production seeding

**Files:**
- Modify: `server/deploy/ensure_default_flow.py`
- Create: `server/tests/test_demo_flow_definitions.py`

- [ ] **Step 1: Write failing flow-definition tests**

```python
def test_ten_step_flow_only_reuses_supported_capabilities():
    steps = ten_step_steps()
    assert len(steps) == 10
    assert [step["type"] for step in steps] == [
        "QR", "FASTENER_MARK", "METER", "FASTENER_MARK", "QR",
        "FASTENER_MARK", "METER", "FASTENER_MARK", "METER", "FASTENER_MARK",
    ]
    assert all(step["required"] and step["require_evidence"] for step in steps)
    assert all(
        step["require_human_confirmation"]
        for step in steps if step["type"] in {"FASTENER_MARK", "METER"}
    )

def test_demo_asset_codes_are_stable():
    assert demo_assignments() == {
        "CRRC-DEMO-001": "CRRC_THREE_STEP",
        "CRRC-DEMO-010": "CRRC_TEN_STEP",
    }
```

Run: `python -m pytest server/tests/test_demo_flow_definitions.py -q`

Expected: import failure because `ten_step_steps` and `demo_assignments` do not exist.

- [ ] **Step 2: Implement ten-step definitions and idempotent assignment creation**

Define ten unique keys in the approved order:

```python
TEN_STEP_KEYS = [
    ("QR_VEHICLE", "QR", "车辆/工位确认"),
    ("FASTENER_A", "FASTENER_MARK", "防松线检查点 A"),
    ("METER_A", "METER", "万用表仪表 A"),
    ("FASTENER_B", "FASTENER_MARK", "防松线检查点 B"),
    ("QR_CABINET", "QR", "设备柜确认"),
    ("FASTENER_C", "FASTENER_MARK", "防松线检查点 C"),
    ("METER_B", "METER", "万用表仪表 B"),
    ("FASTENER_D", "FASTENER_MARK", "防松线检查点 D"),
    ("METER_C", "METER", "万用表仪表 C"),
    ("FASTENER_E", "FASTENER_MARK", "防松线检查点 E"),
]
```

Refactor `main()` to ensure the latest matching three-step and ten-step templates, then list assignments and POST only the missing active demo asset code. Bind both demo assignments to the current admin user for immediate testing. Never print the password or access token. The idempotent selection is:

```python
def demo_assignments() -> dict[str, str]:
    return {
        "CRRC-DEMO-001": "CRRC_THREE_STEP",
        "CRRC-DEMO-010": "CRRC_TEN_STEP",
    }

active = {
    item["asset_code"]: item
    for item in request_json("/api/v1/assignments", token=token)
    if item["status"] not in {"completed", "cancelled"}
}
for asset_code, template_code in demo_assignments().items():
    if asset_code in active:
        continue
    request_json(
        "/api/v1/assignments",
        token=token,
        payload={
            "template_id": latest_templates[template_code]["id"],
            "assignee_id": admin_user["id"],
            "asset_code": asset_code,
        },
    )
```

- [ ] **Step 3: Run server tests**

Run: `python -m pytest server/tests -q`

Expected: all server tests pass.

- [ ] **Step 4: Commit Task 3**

```powershell
git add server/deploy/ensure_default_flow.py server/tests/test_demo_flow_definitions.py
git commit -m "feat(server): seed three and ten step demo tasks"
```

### Task 4: Production and device verification

**Files:**
- Modify: `PROJECT_STATUS.md`
- Create: `docs/validation/2026-09-04-three-and-ten-step-sop.md`

- [ ] **Step 1: Deploy and run the idempotent seeder**

Copy only the updated seeder to `/opt/crrc-sop/deploy/ensure_default_flow.py`, run it in the existing production environment, and run it a second time. The second run must report that templates and active demo assignments already exist, with no duplicate rows.

- [ ] **Step 2: Verify through the public HTTPS API**

Using the server-local password file, authenticate through `https://crrc-glasses.ifix.xin`, then assert:

```python
assert health_status == 200
assert assignments_status == 200
assert {"CRRC-DEMO-001", "CRRC-DEMO-010"} <= {
    item["asset_code"] for item in assignments
}
assert len(by_asset["CRRC-DEMO-001"]["steps"]) == 3
assert len(by_asset["CRRC-DEMO-010"]["steps"]) == 10
```

- [ ] **Step 3: Verify the APK artifact**

Run: `./gradlew.bat clean :app:testDebugUnitTest assembleDebug`

Record test totals, APK byte size, SHA-256, and Git commit in the validation document.

- [ ] **Step 4: Install and exercise the phone when ADB is available**

Run:

```powershell
E:\Android\Sdk\platform-tools\adb.exe devices -l
E:\Android\Sdk\platform-tools\adb.exe install -r app\build\outputs\apk\debug\app-debug.apk
```

On the SOP screen, refresh and verify two task cards, enter both flows, then use one real fastener photo to verify centered ROI, visible box, previous/next persistence, full-image toggle, and the incomplete-review save gate. If no device is attached, record mobile UI verification as blocked rather than passed.

- [ ] **Step 5: Write status and validation evidence, then commit**

```powershell
git add PROJECT_STATUS.md docs/validation/2026-09-04-three-and-ten-step-sop.md
git commit -m "docs: record three and ten step SOP validation"
```
