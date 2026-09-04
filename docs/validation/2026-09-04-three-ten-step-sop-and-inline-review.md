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

## Local verification

- Server: `python -m pytest tests -q` -> 12 passed.
- Android: `:app:testDebugUnitTest` -> 62 tests, 0 failures, 0 errors.
- Android: `:app:assembleDebug` -> PASS.
- Debug APK: 108,792,168 bytes.
- Debug APK SHA-256:
  `90D1F5F88015E03479F8A7F927AEB742978201B15ACE5A978B01662D1BC75FD4`.
- Formal truth `annotations/fastener-v2/instances.json` remained:
  `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`.

ADB was restarted and still reported no connected device. The APK therefore could
not be installed or visually exercised on a phone in this validation run; the build,
unit contracts, and production API are verified, while physical camera interaction
remains pending a USB-debuggable device.

## Accuracy boundary

This UI improves traceability and human review; it does not change the state model's
validated accuracy boundary. The existing 19 historical real ROIs still fail the
strict automatic evidence gate, so the AI state result remains assistive. Production
looseness accuracy must not be claimed until controlled real aligned/displaced pairs
and an independent test set are available.
