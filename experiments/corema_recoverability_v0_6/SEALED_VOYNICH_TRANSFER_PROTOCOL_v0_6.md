# Conditional sealed Voynich transfer protocol — CoReMA v0.6

**Frozen:** 2026-07-25, while the external CoReMA calibration was still running and before its gate outcomes were inspected.  
**Execution condition:** this protocol is executed only if all three external gates in `FROZEN_PROTOCOL_v0_6.md` pass. Otherwise the target remains uninterpreted for this route.

## 1. Objective

Test whether the identity-neutral operational-role structure recovered from real medieval procedural texts transfers to Voynich tokens without using literal word or glyph identity.

A positive result would establish compatibility with the calibrated architecture, not recipe content or any translation.

## 2. Target

Use the existing 37,465-token corpus grouped into 226 complete folios. Preserve manuscript order by line number and token position. Section and scribe metadata are used only for post-prediction stability checks and never as model inputs.

## 3. Representation

Use exactly the structural features frozen for CoReMA:

- token and neighbouring-token lengths;
- recipe/folio-relative position;
- local repetition and equality pattern;
- manuscript/folio-local frequency and frequency rank;
- character diversity and repeated-character indicators;
- sequence-boundary indicators.

Literal CoReMA words, Voynich glyph identities, P70 fields, Currier labels, sections and scribal labels are excluded.

## 4. External calibration of transfer uncertainty

Before scoring Voynich, produce leave-one-manuscript-out CoReMA predictions from the frozen structural-HMM model. From these predictions derive:

1. class-conditional conformal nonconformity scores;
2. robust feature-support distances;
3. manuscript-level role-frequency and transition-signature support regions;
4. expected agreement between raw structural emissions and HMM-smoothed roles.

All thresholds are the empirical 95th percentiles of held-out CoReMA quantities. No target-derived threshold is permitted.

## 5. Target outputs

For every Voynich token report:

- conformal role set;
- point prediction only when the conformal set is a singleton;
- feature-support status;
- raw-versus-HMM agreement.

For each folio and section report:

- singleton coverage;
- role-frequency vector;
- transition signature;
- distance percentile relative to held-out CoReMA manuscripts;
- stability under within-line shuffling and folio-local symbol permutation.

## 6. Frozen adjudication

`OPERATIONAL_ROLE_COMPATIBLE` requires all of:

1. at least 50% of target tokens lie inside the external 95% feature support;
2. at least 35% receive singleton conformal role sets;
3. at least 75% of folios lie inside the 95% CoReMA role-frequency support;
4. at least 75% of folios lie inside the 95% CoReMA transition-signature support;
5. the real-order HMM advantage exceeds within-line shuffling in every section;
6. the sign of every result is unchanged under folio-local symbol permutation.

Failure of conditions 1–4 yields `ABSTAIN_OOD`. Failure only of conditions 5–6 yields `STRUCTURAL_MATCH_WITHOUT_OPERATIONAL_TRANSFER`. No nearest-role account is permitted after either failure.

## 7. Interpretation boundary

Even `OPERATIONAL_ROLE_COMPATIBLE` would not identify ingredients, actions, tools, times, outputs or recipes in Voynich. The labels are calibrated structural roles and may correspond to different domain functions in an unknown technical system.
