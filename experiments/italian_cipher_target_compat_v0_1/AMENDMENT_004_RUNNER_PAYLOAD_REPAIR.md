# Target Compatibility Amendment 004 — runner payload packaging repair

Date: 2026-08-15
Status: **pre-analysis implementation repair**. No external target-stage calibration and no target data were accessed in the failed run described below.

GitHub Actions run `31907037501` failed in the first reconstruction/preflight step with:

- `base64: invalid input`
- `gzip: stdin: invalid compressed data--format violated`

All subsequent steps — external-source download, external calibration, external gate, target download, and target scoring — were skipped. The run therefore produced no scientific target-stage result and exposed no target entropy values.

The failure was traced to corruption/truncation of the single-file committed base64 payload for `run_target_compat_v01.py`. The scientifically frozen local runner itself was unchanged and has SHA-256:

`eda94d458c4c64e2e44c50e7b9799c40efe1e9d0bc064a261963b3e026e19af3`

Its gzip+base64 payload has SHA-256:

`f9dba93894d94576635206ffb4e0f8138507592ea84f8e23d9c70aa130815633`

For robust transport, that exact payload is now stored as five ordered files `runner.part00` ... `runner.part04`. The workflow concatenates the parts, strips only CR/LF transport whitespace, verifies the frozen payload SHA-256, decodes/decompresses it, then verifies the frozen raw-runner SHA-256 before executing any test or data access.

This repair changes no mechanism, source family, entropy metric, representation, parameter, seed, null, threshold, positive-control rule, target-access rule, or verdict logic. It is purely a packaging/reconstruction fix made before any target-stage scientific output.
