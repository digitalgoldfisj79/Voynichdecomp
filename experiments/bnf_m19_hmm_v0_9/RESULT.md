# BnF 7342 free-switch M19 + forward-HMM v0.9 — result

Date: 2026-08-09
Protocol freeze: `e25c149bfdf142cd371f57479e8be7f8b2b9fa65`
Runner: `185f0c55e910ae075dd945402f590586e1bd02cd`
HF job: `6a781da93e1f34a7e32bfe21` — COMPLETED, 511 s running.

## Verdict

**NO M19-HMM SIGNAL under the frozen primary criterion.**

The instrument qualified decisively on a new untouched control partition and therefore legitimately entered the Voynich stage. Voynich produced one notable but sub-threshold result: German was the top held-out language and the only fit with near-perfect independent-key reproducibility, but its forward-likelihood margin over French was 0.02972 nats/letter, below the preregistered 0.05 requirement. No lexical or cross-transcriber candidate gate was triggered.

## Fresh positive-control qualification

Fresh sentence split:
- LM training residues `{3,4,8,9}` mod 10;
- qualification residues `{2,7}`;
- previous development/control residues excluded.

All six full-repertoire controls passed exactly:

| target | correct rank | margin nats/letter | numerical mapping accuracy | independent-fit agreement |
|---|---:|---:|---:|---:|
| Latin | 1 | 0.12293 | 1.0000 | 1.0000 |
| Italian | 1 | 0.15141 | 1.0000 | 1.0000 |
| German | 1 | 0.09008 | 1.0000 | 1.0000 |
| French | 1 | 0.15162 | 1.0000 | 1.0000 |
| Arabic | 1 | 0.17991 | 1.0000 | 1.0000 |
| Spanish | 1 | 0.08665 | 1.0000 | 1.0000 |

Gate: **PASS 6/6**. Minimum margin 0.08665 > 0.05; median and minimum mapping accuracy 1.0; minimum independent-fit agreement 1.0.

Greek and Hebrew remained candidate languages in the Voynich panel but are not full-repertoire synthetic qualification languages under the frozen romanization.

## Voynich census

Primary transcription: ZLZI.

- all pages: 226;
- training sample: 59 folios, 50,941 non-space glyph positions;
- held-out: 45 folios, 31,832 glyph positions;
- all 25 lowercased surface symbols present in training;
- held-out mapping coverage: 100%.

## Voynich held-out ranking

| language | forward nats/letter | independent-fit agreement |
|---|---:|---:|
| **German** | **-2.692293** | **0.991893** |
| French | -2.722013 | 0.543315 |
| Spanish | -2.740437 | 0.607310 |
| Greek | -2.830757 | 0.874207 |
| Latin | -2.848045 | 0.588936 |
| Italian | -2.864642 | 0.506763 |
| Hebrew | -2.887957 | 0.541705 |
| Arabic | -2.996961 | 0.309456 |

German top-v-second margin = **0.0297198 nats/letter**, below the frozen required 0.05. Therefore primary candidate = FALSE.

The stability contrast is noteworthy but non-binding: German's two independent fits agree on 99.19% of training occurrence weight, whereas the next-best forward languages French and Spanish agree only 54.33% and 60.73% respectively. This is not enough to override the preregistered margin gate.

### Frozen German numerical map

The better German fit maps ZLZI surface labels to unmarked BnF numerical values as follows:

`a→5, b→22, c→6, d→4, e→1, f→16, g→22, h→3, i→10, j→20, k→2, l→12, m→9, n→23, o→1, p→7, q→4, r→24, s→30, t→8, u→0, v→28, x→28, y→5, z→20`.

This obeys the exact M19 rule: all 19 values occur, exactly six values are duplicated, and no value has multiplicity >2.

## Interpretation

The qualified negative means the exact global free-switch M19 model does not meet the preregistered evidence standard on ZLZI. However, unlike earlier broad-fit failures, it leaves a specific post-hoc lead worthy of independent testing: the frozen German map is highly reproducible and top-ranked but fails only the language-margin criterion.

Any follow-up test of that German map must be labelled exploratory/post-selection unless a new independent data split is frozen before testing. The v0.9 threshold must not be lowered after the fact.

Implementation note: because `run_v09.py` wraps the v0.8 runner, its emitted `RESULT_JSON` retains the inherited string `"protocol":"v0.8"`. The branch, frozen protocol and actual partition/optimizer are v0.9 as documented above; this is a reporting-label defect only.