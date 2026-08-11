# Amadi Residuals v1 — Sealed H2 Target Result

Date: 2026-08-11
HF target job: `6a7b583627caad61c6eac084`
Executable bundle SHA-256: `a51fe37a1e00a7f4c6189e481d077f0fda62f48d6dd4ff8a5216e1c8f11fe2f4`
Qualification freeze: `QUALIFICATION_FREEZE_V1.json`
Status: **H2 COMPLETE; NO ADMISSIBLE CANDIDATE; C2 REMAINS SEALED**

## Integrity

Target source SHA-256 matched the frozen RF source:
`eb857a1f353b18983fbc25b954e1bbce227a26d99cefabfda9206ff9b57644d2`

Coverage remained `0.9956786919950467`.

H2 comprised 23 previously untouched folios / 15,224 retained characters. C2 comprised 23 untouched folios / 17,256 characters and was not scored.

No plaintext was emitted or inspected.

## Binding H2 results

| family | selected language/rule | H2 score | ABS floor | gap | delta vs M0 | DELTA floor | delta gap | A/B agreement | converged? | binding result |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| R12H | Italian / R12_V1_024 | -2.57705975 | -2.36540723 | -0.21165252 | 0.82571122 | 1.36959196 | -0.54388074 | 0.59725 | **NO** | **UNRESOLVED_SEARCH; NOT A CANDIDATE** |
| VC_END | French / section-013 transform | -3.19702635 | -2.32314223 | -0.87388413 | 0.02077693 | 1.33947524 | -1.31869831 | 1.00000 | **YES** | **CLOSED NEGATIVE / INCOMPATIBLE UNDER V1** |
| PWA | German / K=5 | -2.97247925 | -2.36703209 | -0.60544716 | 0.15518802 | 1.51561428 | -1.36042626 | 0.46851 | **NO** | **UNRESOLVED_SEARCH; NOT A CANDIDATE** |
| GHOUSE5 | Hebrew / 5 selector states | -2.94044947 | -2.63721391 | -0.30323556 | 0.05056180 | 1.80798436 | -1.75742255 | 0.30763 | **NO** | **UNRESOLVED_SEARCH; NOT A CANDIDATE** |

The negative gaps are `H2 score - ABS floor` and `observed delta - DELTA floor`. More negative is worse.

### Critical protocol interpretation

The runner's mechanical fallback label `COMPATIBLE_NONSPECIFIC` for nonconverged R12H/PWA/GHOUSE fits is **not** the binding scientific classification. `PRETARGET_FREEZE_V1.md` and the parent protocol explicitly state that a target fit that fails A/B convergence cannot be used to reject a mechanism. The binding classification is therefore `UNRESOLVED_SEARCH / NO POSITIVE CLAIM`.

Conversely, VC_END converged with A/B agreement 1.0. Its best family fit was dramatically below both the absolute positive-control floor and matched-baseline delta floor. It is therefore a legitimate held-out negative.

## PWA additional gate

Selected target member: `PWA K=5 / German`.

- H2 reset delta: `0.6577524575`
- frozen German reset floor: `1.3925626052`
- reset gap: `-0.7348101477`
- A/B overall agreement: `0.4685051`
- per-state agreement: `[0.53828, 0.47935, 0.50924, 0.38337, 0.35087]`

Thus PWA fails every positive gate visible at H2: absolute compatibility, advantage over M0, word-reset specificity, and map stability. Because the optimizer did not converge, this is **not promoted to a formal family rejection** under v1.

## GHOUSE5 additional gates

Selected target language: Hebrew.

FIT-A payload characters per selector state:
`[43651, 27375, 7870, 2129, 63790]`

All five states therefore pass the frozen minimum support gate (>=500 characters).

However per-state A/B agreements were:
`[0.53447, 0.12493, 0.25502, 0.05636, 0.24570]`

All fail the frozen >=0.90 stability requirement except none; overall agreement was only 0.30763.

The real selector assignment did score above the 99th percentile of 256 within-folio selector-label permutations:

- real H2 score: `-2.9404494720`
- permutation p99: `-3.1151608140`
- permutation maximum: `-3.1057508036`

This permutation result is **non-binding and not evidence of a GHOUSE cipher**, because the fitted maps failed convergence/stability and the family was far below both absolute and matched-baseline positive floors. The preregistered positive interpretation requires all gates jointly.

## R12H

R12H's best Italian target fit was below both frozen floors, but A/B assignment agreement was only `0.59725`, far below the 0.95 convergence/positive requirement. Under the freeze this is unresolved search, not a negative.

## C2 decision

No family produced an admissible H2 candidate.

Therefore:

- exact-source narrowing is not triggered;
- C2 is **not opened**;
- no additional representation, optimizer, language, or rule search is permitted in v1;
- the v1 target programme stops at H2.
