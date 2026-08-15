# ABC/FW recovery — programme synthesis and stopping point

Date: 2026-08-15

## Executive conclusion

The August-13 ABC/FW experiments contained a real result, but the original outputs obscured it.

The evidence does **not** support a single local process generating both adjacent edit-distance-1 (ED1) attraction and exact lag-2 recurrence. It also does **not** support replacing that model with two free-running local kernels.

The best current mechanistic description is:

> **P70-state-dependent line-position architecture.** Exact lag-2 recurrence is concentrated in empty-core structural forms and is largely explained by line-position/LAAFU structure. ED1 attraction is strongly state-preserving and boundary/position coupled, but has no detectable production-order direction and collapses in the trimmed line interior.

This is a production-architecture result, not a decipherment or semantic interpretation.

---

## 1. What the original ABC experiment actually decided

### ABC-A — the two observables are not one varying local process

Recovered result:

- folio Spearman rho = -0.0027, p = 0.968, n = 219;
- historical JSON rho = -0.0112, n = 218;
- section rho = -0.1667, p = 0.668, n = 9.

The preregistered `TWO_PROCESSES` criterion was `|rho| < 0.15` or negative. It is met decisively.

**Consequence:** retire the P2-style architecture in which one local rule is responsible for both ED1 and lag-2 legs.

This does not imply two independent local kernels; later conditioning shows both observables are strongly tied to line architecture.

### ABC-B — short-string crowding is falsified as the VMS ED1 explanation

| mean pair length | observed | null mean | ratio | z |
|---|---:|---:|---:|---:|
| short <=4 | 598 | 483.46 | 1.237 | 6.59 |
| mid 5–6 | 501 | 471.405 | 1.063 | 1.62 |
| long >=7 | 146 | 109.88 | 1.329 | 3.78 |

The preregistered falsification gate was long ratio >=1.15 with z>=2. The observed long class is 1.329, z=3.78.

The reconstructed ReM control validates the intended length discriminator. Across 50 structure-matched diplomatic MHG pseudo-corpora:

- short mean ratio 0.685;
- mid 0.267;
- long 0.204;
- strict short > mid > long gradient in 72%;
- no replicate had any length-class ratio >1.

**Consequence:** R-1/A-series mechanisms may generate edit-neighbour crowding, but the VMS adjacent ED1 excess is not explained by accidental short-word lexical density.

### ABC-C — no production-order arrow

Correctly recomputing the preregistered ratio distributions gives:

- accretion/reduction: observed 0.9453 vs null 1.0029 ± 0.0952, z=-0.61, empirical p=0.537;
- first/second-half substitution site: observed 1.9227 vs null 1.8584 ± 0.2001, z=0.32, empirical p=0.791.

**Consequence:** the adjacent ED1 phenomenon is order-symmetric under this test. Do not describe it as evidence that a scribe systematically copies a previous token and then accretes/reduces it in reading order. This corroborates earlier programme evidence against a copy-arrow.

---

## 2. What FW actually decided

The original function-word hypothesis required all of:

1. lower positional entropy than frequency-matched controls;
2. higher collocational breadth;
3. near-free ABA middle slots.

Historical outputs gave the first two in the opposite direction:

- positional entropy: carriers 3.325 vs controls 3.098;
- breadth: carriers 1.005 vs controls 1.478.

The missing B-slot test is now recovered:

- pooled carrier ABA n=157;
- H(B)=6.684 bits;
- H(B)/maximum=0.916;
- no individual carrier reached the preregistered n>=30 power floor.

Therefore:

- **ordinary function-word reading: falsified** by the original conjunction;
- **fixed repeated-phrase reading: also unsupported**, because B is highly variable rather than fixed;
- the correct description is a structural carrier class with high middle-slot freedom.

---

## 3. P70 explains why the top-20 carrier list was misleading

In the recovery bridge, every recoverable member of the frozen top-20 carrier list is an empty-core P70 form; carrier occurrence empty-core fraction is 1.000 versus 0.527 corpus-wide. Two frozen spellings (`cshedy`, `cshey`) are absent from the recovered P70 occurrence frame and were retained as missing rather than silently renamed.

But carrier identity itself is not the fundamental lag-2 variable:

- top-20 carrier lag-2: ratio 1.156, z=1.86;
- non-carrier lag-2: ratio 1.271, z=3.42;
- empty-core endpoint lag-2: ratio 1.219, z=3.42;
- nonempty-core endpoint lag-2: ratio 1.150, z=0.89.

Empty-core endpoints account for about 92% of the positive lag-2 excess in that bridge.

**Consequence:** stop treating the arbitrary top-20 list as a mechanistic category. It was a frequency-selected proxy for a broader P70 structural state.

---

## 4. The decisive conditioning result

A new follow-up was frozen before calculation and used the exact historical P0 ZLZI / LAAFU null definitions. P70 state mapping covered 100% of the 33,200 P0 token occurrences with no ambiguous token-state mapping.

### Exact lag-2, empty-core endpoints

| null | observed | null mean | ratio | z |
|---|---:|---:|---:|---:|
| N0 whole-line shuffle | 261 | 215.327 | 1.212 | 3.56 |
| N1 boundary-class preserving | 261 | 241.845 | 1.079 | 1.49 |
| N3 trimmed interior under N1 | 136 | 125.514 | 1.084 | 1.12 |

Preregistered verdict: **POSITIONAL_SCAFFOLDING_SUPPORTED**.

The empty-core recurrence signal is real under a naive line null but does not survive the null that preserves first-two / middle / last-two position classes, nor in the trimmed interior.

### ED1, P70 endpoint core-state

N0 reproduces the recovery result: ED1 is preferentially state-preserving.

- both-empty: 1.157, z=4.55;
- mixed: 1.075, z=1.26;
- both-nonempty: 1.372, z=4.58.

Under N1, both same-state classes remain significant:

- both-empty: 1.097, z=3.35;
- both-nonempty: 1.214, z=3.34;
- mixed: 1.051, z=1.08.

But in the N3 trimmed interior they collapse:

- both-empty: 1.049, z=1.20;
- both-nonempty: 1.037, z=0.37;
- mixed: 0.942, z=-0.79.

Preregistered verdict: **BOUNDARY_POSITION_COUPLED**.

**Architecture consequence:** `NO_NEW_LOCAL_KERNELS_YET`.

---

## 5. Revised production architecture

The minimum architecture consistent with the evidence is now:

1. **Section/folio context** selects a vocabulary/slot regime.
2. **Line-position/LAAFU state** changes the emission distribution near line boundaries.
3. **P70 core-state** (empty vs nonempty) is part of that emission regime.
4. These position-conditioned distributions generate:
   - excess exact lag-2 recurrence among empty-core forms;
   - excess adjacent ED1 pairs, preferentially between tokens in the same P70 core-state;
   - without requiring a directional copy operation or an independent lag-2 return operation.
5. The trimmed interior currently supplies no evidence that either observable needs its own free-running local-memory kernel.

This explains why a one-rule local generator was architecturally wrong while also explaining why immediately replacing it with two local rules would be another overfit.

---

## 6. Claims to retire / retain

### Retire

- `ED1 + lag2 are two legs of one local rule`.
- `top-20 ABA carriers are ordinary natural-language function words`.
- `ABA is fixed repeated phraseology`.
- `ED1 is just short-string crowding`.
- `ED1 provides a reading-order copy/accretion/reduction arrow`.
- `raw empty-core lag-2 requires an independent recurrence kernel`.
- `raw ED1 requires a free-running interior edit kernel`.

### Retain

- Adjacent literal ED1 is a genuine VMS structural statistic relative to naive within-line order randomisation.
- Its excess is not a generic consequence of lexical edit-neighbour density.
- ED1 attraction is preferentially P70-state-preserving.
- Exact lag-2 excess is heavily concentrated in empty-core structural forms.
- Both observables are strongly organised by line position/boundary state.
- P70 core-state is mechanistically useful as an operational variable, without assigning it semantic meaning.

---

## 7. Next experiment — one model, no local memory

Do **not** add another copy/repeat generator first.

The next falsifiable model should be a held-out **position-conditioned P70 emission model with zero explicit lookback**.

Nested architecture:

- M0: section + line length only;
- M1: M0 + frozen LAAFU boundary class / relative line position;
- M2: M1 + P70 empty/nonempty core-state and slot-conditioned emission distributions.

Train on folios and evaluate on held-out folios. The model must generate complete lines independently conditional on those states, with no reuse-last, copy, lag-2 return, or edit-neighbour operator.

Primary targets are not generic /84 metrics. They are the exact recovered signatures:

1. raw N0 empty-core E2 excess near 1.21 while N1/N3 residuals remain near null;
2. N0/N1 state-preserving ED1 pattern while N3 remains near null;
3. the long-word ED1 excess pattern, so success cannot come from short-string crowding;
4. no accretion/reduction or substitution-site direction;
5. carrier-like high B-slot freedom without treating the frozen top-20 as a privileged class.

If M2 reproduces these held-out signatures, explicit local copy/repeat machinery is unnecessary for this part of the manuscript. If it fails specifically on one residual after conditioning, that residual becomes the justified target for a local kernel.

## Stopping rule

Do not build a more complex local mechanism until the memoryless position-conditioned model has failed a preregistered held-out target. This is the cleanest way to prevent the programme from returning to generator-by-generator curve fitting.
