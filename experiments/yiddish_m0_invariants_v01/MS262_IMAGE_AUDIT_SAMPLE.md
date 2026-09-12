# MS262 transcription-image audit sample v0.1

Frozen before viewing sampled scans: 2026-09-12.

Selection is independent of all invariant/language scores.

Eligible universe: D0 p001r–p021v page sides with at least one extracted OWY word under the already frozen D0 extractor.

Seed derivation literal string:
`yiddish_m0_invariants_v01_20260912|1e7b9f78ae1b4d2bc3d6c2c593d0d14855bc664f|image_audit_v01`

SHA-256-derived 64-bit seed: `13989793330899824760`.

Python `random.Random(seed).sample(eligible_pages, 6)` yielded the following six page sides, fixed before scan inspection:

- 002r
- 003r
- 004r
- 013v
- 014v
- 018r

Audit rule: compare the complete outer `<ab language="owy">` transcription on each selected side against the corresponding manuscript image. Record literal transcription disagreements, segmentation/markup ambiguities relevant to D0 atom or word boundaries, unreadable/damaged spans, and any image/transcription mismatch. Do not repair text by linguistic plausibility. Editor `<exp>` content is not part of D0 and must be assessed separately where it occurs.

This audit is a source-quality gate only; it cannot qualify language discrimination or authorise Voynich target scoring.
