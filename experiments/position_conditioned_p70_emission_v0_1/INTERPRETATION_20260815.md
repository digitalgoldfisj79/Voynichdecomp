# Position-conditioned P70 emission v0.1 — interpretation / closeout

Date: 2026-08-15

## Decision

The preregistered held-out zero-local-memory architecture **fails**.

All scorer-validation gates passed on the real corpus. The experiment covered 37,465 ZLZI tokens, 5,162 lines and 226 folios, with 5 stratified folio-held-out folds and 30 complete out-of-fold synthetic corpora per model.

| model | F1 | F2 | F3 | F4 | F5 | target log-ratio RMSE |
|---|---|---|---|---|---|---:|
| M0 section+length | fail | fail | fail | pass | pass | 0.1527 |
| M1 + LAAFU position class | fail | fail | fail | pass | pass | 0.1329 |
| M2 + generated P70 state/signature | fail | fail | fail | pass | pass | 0.1216 |

M2 therefore has the frozen verdict `FAILED_MEMORYLESS_ARCHITECTURE` (2/5 fingerprints).

## What M2 does reproduce

It reproduces the two signatures that do not require positive local excess:

- no production-order arrow (F4);
- high ABA middle-slot freedom without a carrier rule (F5; median pooled n=101, H(B)/max=0.945).

Thus those two properties need no dedicated mechanism.

## What it fails to reproduce

### F1 — empty-core lag-2 raw excess

Real: N0/N1/N3 = 1.211 / 1.079 / 1.088.

M2 median: 1.051 / 1.000 / 0.984.

M2 correctly produces a conditioned-null shape but cannot generate the magnitude of the raw lag-2 excess.

### F2 — P70-state-preserving ED1 boundary effect

Real same-state N0 ratios: empty-empty 1.157; nonempty-nonempty 1.375. Under N1 the latter remains 1.213; both collapse in N3 interior.

M2 same-state N0 ratios: 1.052 / 1.103. Under N1 both are ~1.00; N3 is also near null.

Therefore merely changing the independent emission distribution by boundary class and P70 state cannot generate the observed same-state attraction.

### F3 — long-word ED1

Real short/mid/long N0 ratios: 1.244 / 1.063 / 1.323.

M2: 1.046 / 1.064 / 1.073.

The long-word structural excess is almost entirely absent.

## Architecture revision

The preceding state-split result remains valid: the relevant excesses are strongly boundary/position coupled and disappear in the trimmed interior. But the stronger claim that they are generated simply by independent position-conditioned emissions is now rejected.

The minimum live statement is:

> **There is boundary-local dependence between token emissions that is not reducible to section, line length, coarse LAAFU position class, or marginal P70 empty/nonempty state.**

This is narrower than a generic copy/mutate mechanism. The interior remains null under the decisive N3 test, so a free-running whole-line local-memory kernel is still unjustified.

## Smallest next mechanism

The next experiment should target only F1–F3 with a **boundary-pair / boundary-template dependence**, while keeping the line interior memoryless.

Recommended nested test:

- B0 = current M2.
- B1 = distinguish exact boundary positions (`0`, `1`, `n-2`, `n-1`) rather than START/END classes, still independent emissions.
- B2 = emit the first two and final two P70 signatures from learned **joint pair distributions**, with the interior generated independently exactly as M2.
- B3 only if B2 fails = a minimal boundary-local ED1/identity coupling whose strength is fitted on training folios only.

Held-out targets remain F1–F3; F4/F5 are guardrails and may not be degraded. No generic /84 optimisation.

This sequence separates an omitted positional covariate (B1) from genuine pair dependence (B2) before permitting an explicit copy/edit kernel (B3).

## Scientific consequence

The programme has therefore moved from:

`one copy/repeat rule` → rejected,

through `two local kernels` → not justified,

through `independent position-conditioned emissions` → rejected,

to a much narrower live hypothesis:

**structured dependence confined to line-boundary token pairs / boundary templates.**

Cross-transliteration robustness remains required after the mechanism is identified.