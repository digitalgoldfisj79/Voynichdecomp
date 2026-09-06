# Protocol amendment 01: mandatory f116r show-through control

Date added: 2026-08-04

## Reason for amendment

The first completed real-data preflight produced a raw `CANDIDATE_ERASED_TEXT_SIGNAL` result. Visual inspection showed a dense full-page text-like pattern whose layout was compatible with writing on f116r transmitted or reflected through the parchment. The original protocol named show-through as an exclusion but did not implement an explicit opposite-side image control. The raw hidden-text verdict is therefore suspended pending this amendment.

This amendment is corrective, not confirmatory: it was introduced after observing a specific confound. The original result and thresholds remain preserved for audit.

## Frozen recto control

- Obtain an independent public colour image of f116r and record its source URL and SHA-256.
- Segment only visibly dark recto strokes using background subtraction, Otsu thresholding, and conservative connected-component filtering.
- Apply the physically expected backside geometry only: horizontal mirror followed by a clockwise quarter-turn.
- Permit small scale and translation refinement; prohibit arbitrary reflection, free rotation, projective warp, or learned correspondence in this control.
- Dilate the aligned recto support conservatively to accommodate the lower-resolution reference and registration uncertainty.
- Exclude the known right-margin f116v writing from the erased-page test region.
- Measure the fraction of valid f116v candidate pixels explained by aligned recto ink.
- Inspect the recto-independent residual by connected components and a frozen line-likeness rule.

## Revised gate

The raw hidden-text candidate is rejected at preflight resolution when:

1. at least 70% of valid interior candidate pixels fall within aligned recto support; and
2. the remaining components contain no line-like group with area at least 20 pixels and elongation at least 3.0.

The revised verdict is then:

`NO_RECTO_INDEPENDENT_ERASED_TEXT_SIGNAL_AT_PREFLIGHT_RESOLUTION`

Otherwise the verdict is:

`RECTO_CONTROL_INCONCLUSIVE`

## Limitations and next gate

This amendment uses a lower-resolution public colour reference for f116r. A final conclusion requires the matching native-resolution f116r MegaVision spectral cube, registered through the physical sheet geometry. The present control can invalidate an apparent positive; it cannot prove that every residual trace is absent.