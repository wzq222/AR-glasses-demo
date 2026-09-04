# SOP gallery evidence design

## Goal

Allow an operator to complete and test a SOP away from the site by selecting an
existing image for any evidence step. The execution page exposes three sources:

1. capture a new photo;
2. select an image from the phone's system gallery;
3. select an image from the app's existing original-image library.

This feature selects images only. It never deletes or modifies a source image and it
does not accept video files.

## User experience

The existing primary `拍摄本步骤证据` button remains. Two secondary buttons appear
directly below it: `从手机图库选择` and `从原图库选择`. All three are hidden when
the run reaches its submit screen and disabled while analysis or upload is in
progress.

Choosing a phone image opens Android's system document picker with `image/*`.
Choosing an original-library image opens the existing `GalleryActivity` in a new
single-selection mode. Normal gallery browsing remains unchanged outside the SOP.

After selection, the execution page shows the chosen image and runs the current
step's existing QR, witness-line or meter analyzer. Witness-line detections continue
into the same per-point human review UI. The operator saves and uploads the step in
exactly the same way as a newly captured photo.

## Evidence handling

Every selected source is copied into a new file under the app-owned `sop_evidence`
directory before analysis. This creates an immutable run-specific snapshot and keeps
the upload independent from gallery permissions or later source-library cleanup.

The importer will:

- accept JPEG, PNG and WebP image content;
- reject unreadable, empty, larger-than-25-MiB or non-image input with a visible
  error before analysis or upload;
- keep the source image untouched;
- derive the upload media type from validated content rather than assuming JPEG;
- decode with a 2200-pixel maximum analysis edge and apply EXIF orientation before
  analysis while preserving the selected source bytes in the evidence snapshot;
- record `evidence_source` as `CAMERA`, `PHONE_GALLERY` or `APP_GALLERY` in the step
  analysis payload.

Cancelling a picker leaves any previously selected evidence and analysis intact.
A failed import also leaves the existing step state intact. A successful replacement
resets prior overlays and witness decisions before running the new analysis.

## Component changes

- `SopActivity`: register two result launchers, render and enable the two source
  buttons, import selected evidence, and route all sources through one
  preview/analyze method.
- `GalleryActivity`: add an explicit selection mode and return the chosen app-library
  file without changing normal browse mode.
- `SopEvidenceImporter`: validate, snapshot and decode evidence with a small,
  testable API.
- `SopApiClient`: upload the validated file with its actual supported media type.
- `activity_sop.xml`: add the two secondary source buttons beneath capture.

## Validation

Implementation follows red-green TDD for source selection contracts, supported image
types, snapshot naming, upload media types, cancellation safety and UI wiring. The
full Android unit suite and debug build must pass.

The resulting APK will be installed on the connected `NOH-AN01`. A physical-device
test will use existing phone/app-library images to complete `CRRC-DEMO-001` end to
end: start the run, select evidence for all three steps, observe each analyzer,
complete required human review, upload every step, submit the run, and confirm the
completed record in the public management backend. No production claim will be made
from these test images; the run proves workflow integration only.
