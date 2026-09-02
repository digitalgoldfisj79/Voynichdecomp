# VBM v14 — e-family token-frame mediation programme

Date: 2026-09-02
Status: **PREREGISTERED BEFORE V14 OUTPUT**
Parent: V13 closeout `V13_METHOD_NOT_QUALIFIED`.

## Purpose

V11 produced two independent positive structural results:

1. e-ladder nuclei have unusually similar boundary contexts;
2. visible token-boundary halves carry reusable factorised information.

V13 did not validate a global repeated e-operator. V14 therefore asks a different mechanistic question:

> Is the V11 e-ladder similarity largely explained by e-related nuclei occupying the same visible token frames `(L,R)`, or does their contextual similarity persist after conditioning on the same frame?

No plaintext, language model, latent vowel/consonant labels, or global operator is fitted.

## Corpus and parser

Use the unchanged V11/Q0b parser and firewalls:

- ZLZI transcription;
- atoms `ckh, cth, cph, cfh, ch, sh, qo` as multi-glyph left halves;
- token triple `(L,N,R)` with interior nucleus `N`;
- single-glyph tokens have empty nucleus;
- invalid tokens terminate a segment;
- H1/C1 folios remain excluded;
- the SHA256 TRAIN/INTERNAL_HOLDOUT and TRAIN-A/TRAIN-B splits remain exactly those used by V11.

No parser change is permitted after binding output begins.

## Eligible nuclei and e-ladders

Primary nucleus types are non-empty and have >=20 occurrences in full TRAIN.

`e-skeleton(n)` replaces each maximal `e+` run by `E`.

An e-ladder pair is any pair of primary nucleus types with the same e-skeleton and different total e-count. This deliberately reproduces the V11-B primary relation rather than imposing the adjacent-step restriction used in V13.

Primary full-TRAIN inference requires >=20 e-ladder pairs. TRAIN-A and TRAIN-B use the same full-TRAIN eligible type set, exactly as V11-B, to avoid changing the estimand across halves.

## Occurrence representation

For a nucleus occurrence at token i in a valid segment:

- own visible token frame: `F=(L_i,R_i)`;
- external-left half: `X_L = R_(i-1)` or `EDGE`;
- external-right half: `X_R = L_(i+1)` or `EDGE`.

Thus the own token frame is separated from the neighbouring-token environment. This decomposition is fixed before analysis.

## Branch A — e-ladder frame sharing

### Hypothesis

If e-family variants are largely graphotactic alternatives within the same token templates, e-ladder pairs should have more similar distributions over their own visible frames `(L,R)` than matched unrelated nuclei.

### Feature vector

For each nucleus type, construct separately:

- distribution over own left halves L;
- distribution over own right halves R.

Use every TRAIN-attested half category plus OTHER, additive smoothing 0.5, normalise each side separately, concatenate, and use Jensen-Shannon distance.

### Statistic

Median pairwise JS distance over all e-ladder pairs.

### Matched null

10,000 deterministic matched samples. For each e-ladder pair `(a,b)`, replace `b` with an unrelated eligible nucleus matched on:

- occurrence-frequency decile;
- raw nucleus length +/-1;
- e-count +/-1 where possible.

Candidates sharing the same e-skeleton as `a` are forbidden. If no candidate meets all criteria, relax e-count matching first, then length, never frequency decile.

Primary effect is `(null_mean - observed)/null_sd` with empirical one-sided p for unusually low distance.

Repeat independently on TRAIN-A and TRAIN-B using the same full-TRAIN pair list and vocabularies.

### Gate A

`A_E_LADDERS_SHARE_TOKEN_FRAMES` only if:

- >=20 full-TRAIN pairs;
- full z >=2.5 and p<=.01;
- both TRAIN-A and TRAIN-B z>=1.5 in the same direction.

Otherwise `A_NO_STRONG_FRAME_SHARING`.

## Branch B — residual external-context similarity within frame

### Hypothesis

If e-ladder variants remain functionally related beyond simply occupying the same token template, they should have unusually similar neighbouring-token environments even after conditioning on the same own frame `(L,R)`.

### Shared-frame strata

For pair `(a,b)`, consider only own frames F observed for both nucleus types in the relevant split.

A pair is residual-eligible in a split only if:

- at least 2 shared frames exist; and
- `sum_F min(count_a(F),count_b(F)) >= 10`.

For each shared frame F, construct for a and b a smoothed distribution over the two external slots `(X_L,X_R)` separately, using all TRAIN-attested external half categories plus OTHER and EDGE. Compute JS distance between the concatenated distributions.

Pair residual distance is the weighted mean of frame-specific JS distances with weight `min(count_a(F),count_b(F))`.

### Matched residual null

10,000 deterministic samples. For every residual-eligible e-ladder pair, replace b by an unrelated eligible nucleus matched on:

- occurrence-frequency decile;
- raw nucleus length +/-1;
- own-frame overlap mass within +/-0.10 where possible.

Frame-overlap mass is

`sum_F min(count_a(F),count_b(F)) / min(total_a,total_b)`.

Candidates sharing a's e-skeleton are forbidden. Relax overlap tolerance to +/-0.20, then length, if needed; frequency decile is never relaxed.

For each sampled pair calculate the identical conditioned residual distance. A sampled pair failing the residual-eligibility threshold is rejected and redrawn deterministically.

### Gate B

`B_E_RELATION_PERSISTS_WITHIN_FRAME` only if:

- >=15 residual-eligible pairs in full TRAIN;
- full z>=2.5 and empirical p<=.01;
- TRAIN-A and TRAIN-B each have >=8 residual-eligible pairs and z>=1.5 in the same direction.

If full residual pairs <15: `B_UNDERPOWERED_WITHIN_FRAME`.
Otherwise failure: `B_NO_RESIDUAL_E_SIMILARITY`.

## Branch C — held-out incremental e-level information

### Hypothesis

After own frame and e-skeleton are known, e-count may still carry predictive information about the neighbouring-token environment.

Fit smoothed categorical models on TRAIN and score INTERNAL_HOLDOUT occurrences from skeletons that contain at least two eligible e-count levels:

- C0: `P(X_L|F,skeleton)` and `P(X_R|F,skeleton)`;
- C1: `P(X_L|F,skeleton,m)` and `P(X_R|F,skeleton,m)`.

Use hierarchical backoff with Dirichlet pseudocount 1.0 to the corresponding skeleton-only distribution, so unseen fine cells are defined.

Primary statistic:

`DELTA = mean_hold_logp(C1) - mean_hold_logp(C0)`.

### Null

10,000 deterministic skeleton-stratified permutations of e-count labels among nucleus **types** within each eligible skeleton, preserving each type's occurrences and frame/environment records. Refit C1 for each permutation and score the unchanged HOLDOUT observations using the permuted type labels. C0 is invariant.

### Gate C

`C_ECOUNT_HAS_RESIDUAL_PREDICTIVE_INFORMATION` only if:

- at least 10 eligible multi-level skeletons contribute HOLDOUT events;
- DELTA>0;
- DELTA exceeds the 99th percentile null;
- empirical p<=.01.

Otherwise `C_NO_RESIDUAL_ECOUNT_INFORMATION`.

## Programme interpretation

- A pass + B fail + C fail -> `V14_E_LADDER_EFFECT_LARGELY_FRAME_MEDIATED`.
- A pass + (B or C pass) -> `V14_E_RELATION_EXTENDS_BEYOND_TOKEN_FRAME`.
- A fail + (B or C pass) -> `V14_RESIDUAL_E_STRUCTURE_WITHOUT_FRAME_MEDIATION`.
- no branches pass -> `V14_NO_ADDITIONAL_E_MECHANISM_RESOLVED`.

A V14 result does not identify plaintext or a cipher. It only distinguishes token-template mediation from residual e-family structure.

## Evidence rules

- No language/plaintext scoring.
- No post-result parser, frame definition, e-skeleton, matching-rule, or threshold changes.
- Nulls are type/pair based; occurrence count is never treated as independent replication.
- Failed gates cannot be rescued by exploratory variants.
- Negative results are committed.
- CPU only; no GPU is authorised.