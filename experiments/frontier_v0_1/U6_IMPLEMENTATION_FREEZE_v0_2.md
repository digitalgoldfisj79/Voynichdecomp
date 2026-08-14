# U6-v0.2 implementation freeze

Date: 2026-08-14
Status: **FROZEN BEFORE SCRIPTNET DOWNLOAD / MODEL FIT**

This file resolves implementation details left open by `U6_EXTERNAL_INSTRUMENT_PROTOCOL_v0_2.md`; no external result has been calculated yet.

- Maximum admitted local windows per page: **4**. This is within the protocol's prewritten upper bound of 12 and is fixed now for compute tractability. The same cap is used in train, calibration, locked and background-only arms.
- Candidate search per page: 256 SHA-seeded random 96×320 windows after scale normalization; retain the first four meeting the frozen ink-fraction range [0.02,0.35]. Pages yielding zero admitted windows are excluded and counted.
- Connected-component scale estimate: foreground components with area >=3 pixels and height >=2 pixels; use the median height when at least 20 components qualify, otherwise fall back to page height 1200 pixels.
- Scale factor is capped to [0.25,4.0] solely to avoid pathological image allocation; pages hitting a cap are counted in the audit.
- Batch size 64, no gradient accumulation, `num_workers=2`.
- Model selection uses calibration AUC after every epoch 1..5 exactly as preregistered; no early stopping.
- Locked AUC is calculated exactly once from the selected epoch.
- Same-writer positives include every distinct-page pair available inside each writer. The deterministic negative sampler first searches different-writer pages within ±2 admitted windows; if none exists it uses the nearest window-count difference. It produces exactly one negative per positive.
- If ScriptNet's filename convention or image content cannot satisfy the preregistered writer/page/data gates, the run returns a data-gate failure. No filename heuristic beyond the prefix before the first hyphen may be introduced after seeing the data.
