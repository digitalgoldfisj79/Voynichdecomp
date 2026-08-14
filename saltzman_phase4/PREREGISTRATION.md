# ASC ORIGIN MEMORY v0.1 — PREREGISTRATION

Date: 2026-08-14  
Parent Phase 3 commit: `464b83142399017b7400772e34f3ecb1440f0259`  
Parent Phase 3 verdict: `ORIGIN_STATE_WITH_RECURRENCE`  
Protocol SHA-256: `594d6363522225de8ec46c0e3dab23f04b6cdd341578159772a966e7d1e86930`

## Scientific question

Does the replicated Phase 3 gain come from a genuine short-memory latent within-token origin state, and what is its correlation law?

This phase changes **persistence only**. It does not retune the cipher, source language, layout, representation, lineation, or state alphabet.

## Frozen primary architecture

Primary state alphabet is K=2:

- state 0: no cyclic rotation;
- state 1: approximately half-word cyclic rotation (`floor(n/2)` for token length `n`).

The state is global across the 2,000-token document stream and does not reset at artificial line boundaries.

### Model A — fixed blocks

`L = 1, 2, 3, 4, 6, 8, 12`

Arms are `FIXED_RUN{L}_K2`.

### Model B — symmetric two-state Markov memory

`P(stay)=p`, with expected same-state run length `1/(1-p)`:

- M2: p = 1/2
- M3: p = 2/3
- M4: p = 3/4
- M5: p = 4/5
- M8: p = 7/8
- M12: p = 11/12

No post-hoc interpolation is permitted.

## Baselines and controls

- `IDENTITY`
- `OCCURRENCE_K2` — independent binary origin state; exact code/law alias of fixed RUN1 under the same state seed.
- `OCCURRENCE_KALL` — historical Phase 3 occurrence-random comparator only.
- `FIXED_RUN4_K2` — locked architectural positive control from Phase 3.
- `MARKOV_M2_K2` — independent-state law implemented through a different recurrence construction.

Common random numbers are used for cipher plans and scorer permutations across arms. Therefore identical transformed text must score identically; persistence is the changed factor.

## Corpus and scoring invariants

Use the same ReM v2.1 diplomatic MHG builder and Phase 3 scoring payload:

- 190 eligible documents with >=2,000 clean tokens;
- first 2,000 tokens per document;
- groups of 10 tokens for the inherited line-based cipher plan;
- `SWITCH_LINE`;
- ATOMIC and LITERAL;
- robust d3/E1 = worst representation;
- 20 replicates/document;
- 100 scorer permutations/replicate;
- scorer SHA-256 `926da655b603981bc197c248f6dce94fad7b242ab40a89d9d8d69cd40839d6b5`;
- mechanisms SHA-256 `eb5ea3f0d3e8aa2f93301e8870b1198f5dc69a879d53c04dda711011cc440838`.

## Lag-correlation diagnostic

Encode binary state as -1/+1 and compute Pearson autocorrelation at token lags 1 through 24.

Frozen theoretical laws:

- occurrence random: `rho(h)=0`, h>=1;
- fixed block length L: `rho(h)=max(1-h/L,0)`;
- Markov: `rho(h)=(2p-1)^h`.

The empirical state ACF and RMSE to the theoretical curve are reported for every K=2 arm. This diagnostic is mechanistic and is computed independently of the target score.

A direct Voynich phase proxy is **not introduced in v0.1**. The inherited scorer supplies a frozen scalar target phenotype but no observed latent origin-state series. Inventing a new phase proxy after Phase 3 would add a second tunable statistic. Any direct target-side phase proxy therefore requires a separate preregistration.

## Primary endpoints

1. representation-robust d3 improvement versus identity;
2. d3 improvement/retention relative to occurrence-random K=2;
3. representation-robust E1 log-error improvement relative to occurrence-random K=2;
4. agreement of empirical lag-correlation with the arm's frozen theoretical law.

Full gate results are secondary only.

## Preregistered intermediate-persistence test

No best cell is selected post hoc.

For each document:

- Fixed-family middle = mean d3 of RUN3, RUN4, RUN6.
- Fixed-family endpoints = mean d3 of RUN1, RUN12.
- Markov-family middle = mean d3 of M3, M4, M5.
- Markov-family endpoints = mean d3 of M2, M12.

Define:

`C = mean(endpoint d3) - mean(middle d3)`

Positive C means an intermediate-persistence advantage.

The reported estimator is the median paired-document contrast with a fixed-seed 10,000-resample bootstrap 95% CI.

Adjudication:

- `INTERMEDIATE_PERSISTENCE_BOTH`: lower CI > 0 in both families.
- `INTERMEDIATE_PERSISTENCE_PARTIAL`: lower CI > 0 in one family and median C >= 0 in the other.
- `LONGER_PERSISTENCE_TREND`: if the intermediate test fails and >=5/6 fixed adjacent steps plus >=4/5 Markov adjacent steps improve as persistence increases.
- otherwise `BROAD_SHORT_PERSISTENCE_OR_UNRESOLVED`.
- If occurrence K2 or RUN4 K2 fails its preregistered Phase-3-style positive benchmark (median better than identity and >=60% document wins), adjudicate `P3_MECHANISM_NOT_REPLICATED`.

## Freeze / target-separation rule

No `scorer.one_eval` call and no target-derived d3/E1 computation may occur before `protocol.json` hashes to:

`594d6363522225de8ec46c0e3dab23f04b6cdd341578159772a966e7d1e86930`

The GitHub workflow implements this structurally: every scoring shard depends on a separate `freeze` job that verifies the hash and runs only target-free operator/lag self-tests.
