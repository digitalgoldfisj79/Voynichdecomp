# Amadi Residuals v1 — Qualification Result

Date: 2026-08-11
HF job: `6a7b55d627caad61c6eac050`
Executable bundle SHA-256: `a51fe37a1e00a7f4c6189e481d077f0fda62f48d6dd4ff8a5216e1c8f11fe2f4`
Status: **ALL FOUR PRIMARY FAMILIES QUALIFIED; H2 MAY BE OPENED**

## Q0 source fidelity

PASS.

- VC_END implements the explicit prose operation; 28/34 tabulated examples agree mechanically and the six preregistered source discrepancies are exactly the six observed.
- R12 section 024 local-rule examples all pass after the pre-Q1 source correction `g -> deletion`.

## Q1 recovery

Family gates:

| family | result |
|---|---|
| R12H | PASS |
| VC_END | PASS |
| PWA K=2,3,4,5 | PASS |
| GHOUSE5 synthetic state-conditioned maps | PASS |

All binding controls converged. A/B assignment agreement was 1.0 in the formal Q1 rows. R12H recoveries were 1.0000, 0.98418, 1.0000. VC_END recoveries were 1.0000 in the formal rows. PWA and GHOUSE formal recoveries remained comfortably above the frozen minimum; displayed PWA values were mostly 0.985–1.000 and GHOUSE included 0.98753–1.000.

## Q2 blind recognition

PASS.

- R12H recovery gate: PASS
- non-R12 family accuracy: **1.000**
- language accuracy: **1.000**
- PWA exact-rule accuracy: **1.000**
- median recovery: **1.000**
- every language: 3/3 correctly recognized across VC/PWA/GH controls

## Q3 calibration

All four families active: `R12H`, `VC`, `PWA`, `GH`.

All Q3 controls converged. Exact 5th-percentile `ABS_FLOOR`, matched-baseline `DELTA_FLOOR`, and PWA `RESET_DELTA_FLOOR` values are frozen in `QUALIFICATION_FREEZE_V1.json` and are binding for H2.

PWA diagnostic caveat: in some Q3 controls generated under true K=2, K=4 tied/outscored K=2 because K=4 contains a K=2 periodic solution. This is a nested-model identifiability issue, not a recovery failure. It does not alter the frozen family or target decision rule. If PWA survives H2, exact-source narrowing remains mandatory before C2.

## Q4 structured negatives

PASS with a clean result:

- total false positives: **0/80**
- iid: 0/16
- order-2 Markov: 0/16
- motif repeat/mutate: 0/16
- copy/mutate: 0/16
- slot grammar: 0/16

Thus the qualified instruments distinguish fresh positives from all five preregistered structured-negative families under the frozen gates.

## Decision

The qualification stage does not support any statement about Voynich because H2 has not yet been scored. It establishes only that the four admitted mechanisms are recoverable/recognizable enough for a legitimate held-out compatibility test.

C2 remains sealed.