# LF-T1 label-free cross-quire transfer — REAL PILOT, 2026-10-19

## RETRACTED / NON-PROMOTABLE FINDINGS
- Previous required botanical-naming gate is WITHDRAWN (see NO_PLANT_NAME_GATE_CORRECTION_20261019.md).
- Any proposition that an improvement in text prediction entails source meaning or lexical structure is WITHDRAWN. The IDENT-GATE exact source/production twins remain indistinguishable. This experiment does not restore source identifiability.
- This pilot does **not** qualify the historical Voynich inverse decoder: Q1B remains failed.
- Do not treat +0.005581 bits/token as an unexplained source effect. Its **matched-null mean is +0.004426** and only the difference is potentially new; that excess is unresolved.

## Prereg and data
[Protocol](LABEL_FREE_TRANSFER_PREREG_20261019.md) was committed as `d0cd2f952d00843548b12971b815904eb5781cd0` BEFORE real-text scoring. [Export query](lf_transfer_export_20261019.sql), read-only research-agent Supabase project.

Dataset: `public.vms_directional_token_v01` joined to `vms_catalogue.vms_folios`, unique hand-1/Currier-A **Herbal-A running** only. 4,580 non-null source token occurrences, 80 folio sides, 7 independent quire fold IDs A–G. Each page's first 15 tokens provided the permitted prefix context (not scored), leaving 3,389 held-out target tokens (73.9956% of corpus). All pages in one quire withheld together; no physical bifolium was split. All other sections/registers excluded (including Herbal-B leaves in quire O/Q).

## Actual outcome
Models use a train-fold top-96 exact word vocabulary plus OTHER tail, with M0 context and M1 online exact/one-edit copying cache, and M2 training-quire early-to-late word association. No plant names, image labels or dictionaries.
- Mean held-out cross-entropy M0: 4.0586901 bits/token.
- Mean held-out cross-entropy M1: 4.0144899 bits/token.
- Mean held-out cross-entropy M2: 4.0089088 bits/token.
- Observed M2–M1 heldout gain: +0.00558105 bits/token.
- Training-document late-block permutation within quire and sorted groups of up to four length-matched pages: 199 permutations, seeded PRNG 20261019.
- Matched-null gain mean: +0.00442607 bits/token, null SD **0.00209532 bits/token**.
- Difference beyond matched null: **+0.00115499 bits/token**, z=**0.55122**, permutation upper p=**0.31** (inclusive add-one correction).
- Permutation null 5th, 50th, 95th percentiles: +0.00149580, +0.00429476, +0.00798064 bits/token.
- Six out of seven quires positive unadjusted, but prespecified gate requires raw gain ≥0.02 bits/token AND p≤0.05 AND ≥5/7 positive. **FAIL.**
- Within-quire page-block permutation is a partial hard negative: it preserves training production blocks and quire, but does not preserve early-late copying-template links; this limitation can only bias against strong content inference.

### Independent whole-quire holdout results
| Quire | Test sides | Scored tokens | M2–M1 bits/token |
|---|---:|---:|---:|
| A | 13 | 430 | +0.002740 |
| B | 12 | 446 | +0.013110 |
| C | 16 | 918 | +0.009456 |
| D | 11 | 296 | +0.003678 |
| E | 8 | 291 | +0.022042 |
| F | 8 | 347 | +0.006875 |
| G | 12 | 661 | **−0.010107** |

Bounding test: leave-one-quire-out aggregate unadjusted gain ranges +0.004035 (omit E) to +0.009382 bits/token (omit G). Even omitting the best quire does not reverse the sign; none of this is a valid post-hoc confirmatory subgroup or a new null test. The original 199 null scores must be recalculated under each new population before any such inference.

## Decision / audit
**The metric does not resolve this.** The positive point estimate differs from null by less than one null SD. There is no confirmatory transfer effect beyond the matching production null. The correct status is `PILOT_FAIL_NOT_IDENTIFIABLE`. An apparent 6/7 positive pattern does not override the aggregated matched null.

Before another model escalation: (1) audit the token table's original RF transcription provenance and independently reconstruct the corpus from ZLZI/TTLI; (2) qualify exact FORM+CONNECT+LINE_ENTRY rendering and a stronger cross-quire copy-template model of matched capacity; (3) rerun the frozen decision on data kept genuinely sealed after calibration. This pilot itself cannot discriminate semantics from copying even if future gains cross threshold.
