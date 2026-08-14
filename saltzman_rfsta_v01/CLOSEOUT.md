# VOYNICH-RF-STA-REPLAY-v0.1 — closeout

Date: 2026-08-14

## Status
Completed successfully on GitHub Actions run `31806441650`, final successful workflow head `f5d0d2223fd9c621e35ec9f98ae9afe27e6aaad2`.

Frozen protocol SHA-256: `8952d3d7990a349e3f8b6cbce4786a69067ace95bed6e07ef935efb072741592`.

Final artifact: `asc-rf-sta-replay-v01-final`, artifact ID `9221416862`, digest `sha256:edbeb09a3ad4ae70102de6fe56bc8012f9a3cb7bb1058b7bd1833c0d494a2df1`.

## Why this rerun was required
The upstream ASC source-generalisation protocol made representation robustness binding: later Voynich comparison had to survive full RF/STA-member, STA-family, and connected-`aaa` projections, and could not rely on surface ASCII decomposition. Phase 9 v0.1 instead used `Paper/Cipher_paper/enriched_records.pkl` and the legacy 85-metric battery. That result is retained as an auxiliary consequence test but is not a binding representation-corrected Voynich test.

This v0.1 replay restored the upstream representation requirement. No synthetic mechanism parameter was retuned and no ReM simulation was rerun. The exact 4,940 Phase-8 cells from workflow run `31800905368` were reused.

## Provenance gates
All passed before target scoring:

- RF1b SHA-256 `81c331b7d8e76761e27d350c3b37ccfbe192848e6c8a227bcb5d40fb29259b17`
- `bitrans.c` SHA-256 `3ffc7e6c74078f9b395179aaf5daaae3c8dfbc2896d21162c8ff0354108e9a`
- `STA-aaa.bit` SHA-256 `622621463ff2973ff456b02f0b46ba99fef8ad9103c464e44427762863e3cb64`
- regenerated RF1b `aaa` SHA-256 `c14f43c731f46274f35b604356c6bb96a1186e0836aa9aa2b518666cce854167`
- unchanged ED1 scorer SHA-256 `926da655b603981bc197c248f6dce94fad7b242ab40a89d9d8d69cd40839d6b5`

Parsing retained 37,193 clean words in 6,515 clean adjacency segments from 5,385 RF text-line records; 655 uncertain/non-clean words were excluded as hard breaks, and 761 `<->` drawing interruptions were hard breaks.

RF-member/full STA, STA-family, and connected-`aaa` are correlated projections, not independent replications.

## Representation-specific Voynich Q targets
Order: `[ED1_N0, ED1_N1, ED1_N3, E1_N0]`.

| representation | median Q4 | median structural gate | target-seed structural passes |
|---|---|---:|---:|
| RF_MEMBER | `[1.138399, 1.101283, 1.057774, 1.008309]` | PASS | 20/20 = 100% |
| STA_FAMILY | `[1.112071, 1.091124, 1.049487, 1.099793]` | PASS | 11/20 = 55% |
| AAA_CONNECTED | `[1.120210, 1.083575, 1.031871, 1.021654]` | PASS | 20/20 = 100% |

All three medians retain the qualitative/median structural attenuation pattern: ED1_N0 > ED1_N1 > ED1_N3, ED1_N0 > 1, ED1_N3 near neutral, and E1 near neutral. However, the preregistered binding robustness rule required >=80% seed-level structural passes in every representation. STA_FAMILY achieved only 55%.

Therefore the binding target-robustness gate FAILS.

## Frozen Phase-8 mechanism replay
Baseline: `CIPHER_ONLY`.
Canonical mechanism: `FIXED_LINE_RESET__POST`.
Sensitivity: `FIXED_CONTINUOUS__POST`.

For each Voynich representation and document, distance is robust across ATOMIC/LITERAL synthetic renderings by taking the worse d3.

| VMS representation | cipher-only median d3 | canonical median d3 | paired median gain | bootstrap 95% CI |
|---|---:|---:|---:|---:|
| RF_MEMBER | 0.938037 | 0.583014 | +0.333689 | [0.306148, 0.367995] |
| STA_FAMILY | 0.913443 | 0.559546 | +0.332272 | [0.294932, 0.367549] |
| AAA_CONNECTED | 0.906232 | 0.551986 | +0.332980 | [0.293504, 0.363456] |

Thus the already-frozen origin-state mechanism moves the synthetic ReM ciphertext toward the Voynich ED1 phenotype in the same direction and by essentially the same amount in all three manuscript-functional projections. This directional effect is representation-robust.

However, it does not come remotely close to the preregistered match threshold d3 <= 0.15. Canonical medians remain 0.552–0.583.

The continuous-state sensitivity is essentially the same: median d3 gains are +0.335465 RF, +0.335663 STA-family, and +0.335212 connected-aaa.

## E1 / full-Q secondary
The mechanism also improves finite E1/full-Q errors relative to cipher-only, but absolute fit remains poor and coverage is incomplete because many ReM documents have zero/nonpositive E1 estimates. Canonical finite median d4 is 1.856 RF, 1.933 STA-family, and 1.864 connected-aaa; these are far above 0.15.

## Preregistered adjudication
`VMS_Q_NOT_REPRESENTATION_ROBUST`

This label is forced by the STA-family seed-stability failure and cannot be rescued by the robust directional mechanism gain.

## Scientific interpretation
Two distinct findings should not be conflated:

1. **Voynich measurement:** the median local-neighbour attenuation shape is surprisingly stable across RF-member, STA-family and connected-aaa, but the strict seed-level structural gate is not stable at STA-family scale because the Q4 vector sits close to the E1 boundary and only 11/20 target-null seeds pass all structural conditions.
2. **Mechanism transfer:** the short-lived origin-state mechanism discovered on ReM produces a large and extremely consistent directional improvement toward all three representation-specific Voynich ED1 targets (~+0.33 d3), without any retuning. This is a genuine representation-robust mechanistic effect, but it remains much too far from the Voynich target to constitute a match.

Therefore the correct claim is narrower than either the old positive-looking ED1 story or the legacy Phase-9 closure: **the origin-state mechanism is a representation-robust partial mimic of the Voynich ED1 attenuation phenotype, not a representation-robust reproduction of the Voynich phenotype.**

Under the frozen stopping rule no mechanism retuning follows this result. A broad held-out Voynich consequence panel is not promoted to a confirmatory next stage from this branch because the binding target-robustness gate failed and the canonical d3 match threshold failed decisively.
