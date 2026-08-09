# BnF M19 STA Identifiability v1.8 — Result

Date: 2026-08-09
Branch: `experiment/bnf-m19-sta-identifiability-v1.8-20260809`
Frozen protocol commit: `ce17c70e858aa5676a66a9a13880a750931e0101`
Diagnostic runner: `fe8acdac518218cb9124e74aea845cc58a0a0668`
HF job: `6a7866fd3e1f34a7e32c040b`
Scope: control-only; no Voynich RF/H17/C17 score was generated.

## Exact reproduction

The binding v1.7 Q3 Arabic K=22 control was reproduced exactly:

- mapping recovery: **0.7632051282051282**
- independent-fit agreement: **1.0**
- Q3 fitted objective: **-1.5656046025840804**
- true hidden-map objective: **-1.5633189510272494**

The true map is therefore better than the converged Q3 fitted map by **0.0022856515568310 nats/scored event**, far above the prospective materiality threshold `1e-5`.

The Q3 error is a coherent exchange of BnF numerical states 1 and 4 across three high-frequency opaque surfaces:

- S01: true 1, fitted 4; holdout frequency 4347
- S03: true 4, fitted 1; holdout frequency 2410
- S04: true 4, fitted 1; holdout frequency 2478

This explains the low occurrence-weighted exact recovery despite perfect agreement between the two original optimizer runs: both runs converged to the same inferior basin.

## D1 stronger search

The preregistered stronger search used 24 independent restarts × 100,000 proposals plus exhaustive legal local polish.

Best result:

- objective: **-1.5633189389232518**
- improvement over Q3 fit: **+0.0022856636608286 nats/event**
- exact occurrence-weighted recovery: **0.966**
- difference from the true-map objective: **+1.2104e-8 nats/event** (numerically negligible and slightly sample-favourable)

The best basin was independently found on restarts 8, 11, 19 and 23. The residual 3.4% exact-map discrepancy is a different, tiny swap involving a much closer objective degeneracy and is not the source of the original Q3 failure.

Formal D1 classification: **OPTIMIZER MISS**.

## D2 state geometry

There are **no exact single-transposition automorphisms** of the frozen 19-state Arabic induced model at tolerance `1e-12`.

The Q3-confused numerical pair (BnF values 1 and 4) is relatively close but not equivalent:

- standardized signature-distance rank: **10th of 171 pairs**
- percentile: **5.85%**
- cosine similarity: **0.85057**

Thus the original error is understandable as a difficult local basin involving two fairly similar numerical states, but it is not a structural symmetry of the model.

The closest state pairs overall include values 9/16, 3/8, 2/8 and 5/23. The latter is relevant to the tiny residual difference between the strongest-search map and the exact hidden map.

## Verdict

**OPTIMIZER MISS**

The v1.7 K=22 Arabic qualification failure was caused by insufficient global optimization, not by intrinsic non-identifiability of the Arabic M19 channel. The frozen true map has materially higher objective than the Q3 fitted map, and stronger search reaches its objective basin repeatedly.

D3/D4 large-sample equivalence testing is therefore not invoked as the primary explanation under the frozen v1.8 protocol. The correct next experiment is a fresh, prospectively frozen qualification using a stronger convergence-controlled optimizer while retaining the original map-recovery thresholds.

No v1.7 Voynich result is retroactively unlocked by this diagnosis.
