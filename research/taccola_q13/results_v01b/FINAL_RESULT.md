# Taccola × Q13 calibration v0.1b — FINAL

## RETRACTED FINDINGS

- **RETRACTED:** the raw Siena/Taccola concentration (4/12 vs 5/185) is not independent evidence. After family collapse it is 1/12 vs 5/185; effect +5.63 percentage points, exact conditional null SD 5.13 points, 1.10 null SD, Fisher p=0.318. The metric does not resolve this.

## Frozen calibration outcome

Status: **FAILED CALIBRATION**. Q13 remains sealed.

Panel SHA-256: `8226f0435ee07d8af1e4b0d8cb4a8f09af8f7a82b32bf04441f8dab7ae49c905`

Original scientific payload SHA-256: `59d4a2635b5a64de32a1fd69c577c123fab1fe81bb1739a632ec7035dc5b4f5b`

Final workflow run: `33300853008`

Final result artifact: `taccola-q13-complete-calibration-v01b`, artifact ID `9728903636`, artifact SHA-256 `6bfa0076d8f67a5080727a42c45b2cc05f9a2d2365c74a9754fc22f327d0a690`.

All 37 manuscript blocks completed. Active null blocks: 34. Missing required witnesses: 0. Image errors: 0. Manifest errors: 0. Palatino 766 and LJS 419 used documented pre-score transport-only r15 fallbacks; sampling and all scientific scoring functions remained frozen.

### Clm 197 II → Palatino 766

- Composite: effect +0.5111; null SD 0.8416; z=0.6073; empirical block p=0.3143; technical rank 4/7; bootstrap fraction z≥2 = 0.000.
- HOG: effect +0.0019; z=0.0233; p=0.4571.
- Chamfer: effect −0.0430; z=−0.8899; p=0.8000.
- Geometry: effect +0.1949; z=2.3998; p=0.0286; overall rank 1/35.

### Palatino 766 → Clm 197 II

- Composite: effect +0.9743; null SD 0.8238; z=1.1827; empirical block p=0.1429; technical rank 1/7; bootstrap fraction z≥2 = 0.000.
- HOG: effect +0.0446; z=0.7001; p=0.3143.
- Chamfer: effect −0.0025; z=−0.0752; p=0.5429.
- Geometry: effect +0.1879; z=2.2981; p=0.0286; overall rank 1/35.

### Frozen gates

- Passing representations in both directions: `geometry` only.
- At least two passing representations: **FAIL**.
- Non-degenerate passing representation pair: **FAIL**.
- Composite gate: **FAIL**.
- Technical-rank gate: **FAIL**.
- Decision-rule fragility gate: **FAIL**.
- Calibration passed: **FALSE**.
- Q13 unseal allowed: **FALSE**.

## Interpretation

The v0.1 whole-page instrument does not establish a Taccola-specific visual signature. Geometry alone distinguishes the autograph pair in both directions, but HOG and chamfer do not. Several generic BSB/background manuscripts outrank the positive pair under HOG/chamfer, consistent with those representations capturing page/document layout rather than Taccola-specific motifs.

This result is closed. v0.1/v0.1b thresholds must not be relaxed post hoc. Any successor method must be developed using the exposed v0.1 data and validated on genuinely unseen Taccola-family witnesses before Q13 can be unsealed.
