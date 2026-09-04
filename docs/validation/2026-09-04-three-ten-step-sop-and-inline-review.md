# Three/ten-step SOP and inline witness review validation

## Outcome

The production SOP service now exposes two active demo assignments to the existing
`admin` account:

- `CRRC-DEMO-001`: the existing three-step QR, witness-mark, and meter flow.
- `CRRC-DEMO-010`: a ten-step flow interleaving QR, five witness-mark checks, and
  three meter checks.

The Android `FASTENER_MARK` step now reviews one detected point at a time inside the
same SOP step. Each view is cropped from the decoded source photo at 2.25 times the
detected target's longest side, retains neutral padding at image boundaries so the
target stays centered, and draws the original ROI explicitly. The operator can move
between points or open the original photo with only the current ROI highlighted.
All point decisions must be recorded before the step can be saved. A target shorter
than 32 source pixels defaults to `unable to judge / recapture` and prompts a closer
photo.

## Production verification

The idempotent seeder was run twice on `101.200.152.104`:

- First run reused `CRRC_THREE_STEP` v2 and `CRRC-DEMO-001`, then created
  `CRRC_TEN_STEP` v1 and `CRRC-DEMO-010`.
- Second run reused both templates and both active assignments without creating a
  duplicate.
- An authenticated smoke test against `https://crrc-glasses.ifix.xin` passed login,
  template listing, assignment listing, the three-step length, and the exact
  ten-step type order.

No credential or access token was printed or stored in this repository.

A later live authentication check used credentials supplied out of band and retained
no password or token. The physical phone logged in through the production HTTPS API
and displayed exactly two pending assignments: `CRRC-DEMO-010` (ten steps) and
`CRRC-DEMO-001` (three steps). An independent Playwright browser session also logged
into the public management site, showed one active user, three SOP versions and two
pending assignments, and expanded `CRRC_TEN_STEP` as 10 steps with five witness-line
review points. The only browser console error was a non-functional `/favicon.ico`
404; authentication and business data requests succeeded. The browser session was
logged out and closed after verification.

## Local verification

- Server: `python -m pytest tests -q` -> 12 passed.
- Android: `:app:testDebugUnitTest` -> 63 tests, 0 failures, 0 errors.
- Android: `:app:assembleDebug` -> PASS.
- Debug APK: 108,792,293 bytes.
- Debug APK SHA-256:
  `AFD477F93735EF9888BB96888A1F9F35EC6B6198B4199AFF492C6ED2E6BBE953`.
- Formal truth `annotations/fastener-v2/instances.json` remained:
  `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`.

At the start of this validation, restarting ADB still reported no connected device,
so initial checks were limited to the build, unit contracts and production API. The
later physical-device recovery and installation result is recorded below.

An isolated Android 16/API 36 x86_64 emulator with ARM64 translation was then used as
a local startup check. The first run exposed a pre-existing crash when an audio-less
device rejected `ToneGenerator` initialization. `VoiceController` now treats the
prompt tone as optional, and a regression contract covers the fallback. After the
fix, the APK installed with `primaryCpuAbi=arm64-v8a`, cold-launched without a fatal
exception, kept its process alive, rendered the main “中车巡检任务” card, and opened
the SOP login screen. The physical Huawei connection initially disappeared from the
USB bus every 5-8 seconds and also produced descriptor error code 43. After replacing
the cable, the same `NOH-AN01` composite device remained continuously present for the
30-second observation window, isolating the remaining failure to its 2016 Huawei
`ew_usbccgpfilter` driver: Windows reports `CM_PROB_FAILED_START` / code 10 before an
ADB child interface can enumerate. The user reported that the APK was installed
manually on the physical phone; package identity, startup logs, and SOP interaction on
that phone initially remained unverified. A subsequent known-good data cable allowed
the same `NOH-AN01` to enumerate normally through the Microsoft ADB interface. The
latest APK was then installed with `adb install -r -g`, cold-launched in 2.342 seconds,
and kept its process alive. UIAutomator confirmed the main inspection entry and opened
`SopActivity`; the “中车巡检工作台” login title, username field, and password field
were all present, with zero matching fatal AndroidRuntime entries. No device serial is
stored in this repository.

## Accuracy boundary

This UI improves traceability and human review; it does not change the state model's
validated accuracy boundary. The existing 19 historical real ROIs still fail the
strict automatic evidence gate, so the AI state result remains assistive. Production
looseness accuracy must not be claimed until controlled real aligned/displaced pairs
and an independent test set are available.
