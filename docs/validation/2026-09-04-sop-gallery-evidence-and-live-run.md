# SOP gallery evidence and physical-device validation

## Outcome

Every SOP step now offers three image evidence sources: camera capture, phone gallery,
and App original gallery. A selected gallery image is copied into an app-owned evidence
snapshot before analysis. The original source is never deleted or modified. Successful
replacement resets the prior analysis and witness review; cancellation, unsupported
type, oversized input, and decode failure preserve the existing evidence.

Snapshots accept JPEG, PNG, and WebP up to 25 MiB. Decode is bounded to a 2200-pixel
long edge and applies EXIF orientation. Evidence uploads use the file's actual media
type, and step `value_json` records `CAMERA`, `PHONE_GALLERY`, or `APP_GALLERY` as
`evidence_source`.

## Original gallery assets

Three user-supplied archives were validated for path traversal and extracted outside
Git to:

`E:\Work\京新数智\识动hicool\中车眼镜数据资产\app-original-gallery`

The physical phone's App original gallery contains one QR image, one witness-line
challenge image, and five meter images from those archives. An earlier manually
imported photo remains present, for eight images total. No case image or archive was
added to this repository or APK.

## Physical-device run

The debug APK was installed on the connected Huawei phone. UIAutomator verified the
three source buttons, an initially empty App original gallery, import from the system
picker, later display of eight App-original images, and normal return from selection
mode into the same SOP step.

One three-step production run completed and was submitted with:

- QR step: `PHONE_GALLERY`, JPEG evidence, 3,093,011 bytes.
- Witness-line step: `APP_GALLERY`, JPEG evidence, 121,805 bytes.
- Meter step: `APP_GALLERY`, JPEG evidence, 165,567 bytes.

The production database reported the run as `submitted`, with all three step results
and evidence records present. A queue-state defect discovered during this run was
fixed and deployed: an assignment with a submitted run is now returned as `submitted`,
excluded from mobile pending work and the dashboard pending count, and cannot start a
duplicate run. Production verification showed one pending assignment, one awaiting
review, and HTTP 409 for a duplicate start. The updated phone displayed only the
remaining ten-step task.

## Accuracy boundary exposed by the cases

This run validates transport, UI, evidence preservation, upload, and workflow state;
it does not validate model accuracy. The supplied witness-line challenge image produced
zero marked-point candidates, and the selected 233.5 V meter image produced no stable
reading. Both steps were explicitly submitted as `unable_to_judge` for human review.
The archives do not provide witness-line normal/loose ground truth, so the zero-result
case must not be relabeled or used as an automatic accuracy claim.

## Verification

- Android unit tests: 70 passed, 0 failed, 0 errors.
- Android debug APK build: passed.
- Server tests: 12 passed.
- APK size: 109,979,746 bytes.
- APK SHA-256: `708519B0D8181CF8D3F495A96CB6E9572EFF7DE8E211D77ABE9F986388E4E5F7`.
- Deployed `server/app/main.py` SHA-256:
  `4126F853FECE50BA311216FDD2AFD3411479C6E432EFD55BE388608E09F612A0`.
- Production health endpoint: passed.
- Fatal AndroidRuntime entries during the final phone check: zero.
- Formal truth SHA-256 remained:
  `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`.
