# STA boundary-return discriminator v0.1 — build status

Date: 2026-08-15
Branch: `experiment/sta-boundary-return-discriminator-v0.1-20260815`

Status: **BUILT / TARGET NOT RUN**.

The push-triggered GitHub Actions build check completed successfully (run `31878194315`). The workflow compiled both executables and ran only the synthetic target-blind build check. Canonical RF1b acquisition and the full discriminator steps were skipped by design; the full target run is `workflow_dispatch` only.

Synthetic build-check output:

- unplanted fitted boundary q: `0.0000`;
- planted synthetic boundary-return fitted q: `0.1125`;
- unplanted E2 N0 ratio: `1.02336`;
- planted E2 N0 ratio: `1.50473`;
- build verdict: `BUILD_CHECK_PASS`.

These values are engineering checks only and are not Voynich target results. No B0/B1/B2/B3 target model output has been generated or inspected on this branch.

Frozen scientific artefacts:

- `PREREG_20260815.md`;
- `IMPLEMENTATION_FREEZE_20260815.md`;
- `AMENDMENT_01_PRETARGET_20260815.md`;
- `run_discriminator.py`;
- `run_build_checks.py`;
- `.github/workflows/sta-boundary-return-discriminator-v0.1.yml`.
