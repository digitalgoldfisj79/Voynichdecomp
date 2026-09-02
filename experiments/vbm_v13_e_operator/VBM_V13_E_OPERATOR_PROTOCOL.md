# VBM v13 — Shared e-Operator Ciphertext Test

Date: 2026-09-02
Status: **PREREGISTERED BEFORE V13 OUTPUT**
Parent: V12 closeout `V12_COMPOSITIONAL_TRANSDUCER_FAILS_PRESSURE_TEST` and post-closeout continuation boundary.

## Purpose

V11 established that nucleus types sharing an e-skeleton but differing in e multiplicity occupy unusually similar boundary contexts. V12 showed that one specific global-permutation formalisation of that observation was highly recoverable at compact pressure but failed its stronger Stage-B frequent-type gate on the nucleus side.

V13 does not rescue V12 and does not propose a new decipherment. It asks the narrower ciphertext-only question required by the V12 closeout:

> Does adding one `e` behave like a shared repeated transformation of nucleus boundary-context distributions across different e-skeletons, or are e-ladder members merely similar in a skeleton-specific way?

A positive answer is required before any smaller nucleus transducer may be proposed.

## Forbidden in V13

- no plaintext or language likelihood;
- no vowel assumption;
- no consonant or consonant-length interpretation;
- no V12 permutation operator fit to Voynich;
- no bridge-value assignment;
- no H1/C1 material;
- no parser changes.

## Corpus and parser

Use the unchanged Joachim-exact Q0b parser and the same exclusions and deterministic folio split as V11:

- ZLZI transcription from `voynich_transcriptions_slim.json`;
- left atoms `ckh, cth, cph, cfh, ch, sh, qo`, otherwise first glyph;
- right half = final glyph;
- nucleus = token interior;
- single-glyph token = shared left/right half with empty nucleus;
- bridge = `right(previous)|left(next)`;
- invalid tokens terminate a segment;
- H1 and C1 folios remain excluded;
- folios split by the unchanged SHA256 Q0b TRAIN / INTERNAL_HOLDOUT rule.

## Eligible adjacent e-ladders

For every non-empty nucleus type in TRAIN:

- `e-skeleton` = replace each maximal `e+` run by one `E`;
- `m` = total count of glyph `e` in the nucleus.

A primary ladder edge is an ordered pair `(n_m, n_{m+1})` such that:

- same e-skeleton;
- e-count differs by exactly +1;
- both types have >=20 TRAIN occurrences.

For primary untouched evaluation, both members must additionally have >=5 INTERNAL_HOLDOUT occurrences.

The inferential unit is the e-skeleton, not the nucleus occurrence or ladder edge.

## Boundary-context representation

Freeze the 32 most frequent TRAIN bridge types plus `OTHER`. Build separate preceding-bridge and following-bridge count vectors for every eligible nucleus type.

Each probability vector receives additive smoothing 0.5 and is normalised separately for preceding and following contexts.

No position, section, hand, language, or plaintext feature is included.

## Shared multiplicative operator

For one ordered ladder edge, define separately for preceding and following contexts:

`r_j = log(p_{m+1,j} / p_{m,j})`.

For a set of training edges, the shared operator is the coordinate-wise median of `r`, with each side centred by subtracting its arithmetic mean and clipped to `[-2,2]`.

Applying an operator `w` to a source distribution `p` gives

`T_w(p)_j proportional p_j * exp(w_j)`.

This is an abstract transformation of context distributions. It is not a letter, state, vowel, or plaintext mapping.

## Primary cross-skeleton / held-out prediction

For each untouched-evaluable ladder edge belonging to skeleton `s`:

1. estimate `w_-s` from all eligible TRAIN ladder edges whose skeleton is not `s`;
2. construct the source distribution from INTERNAL_HOLDOUT counts of `n_m`;
3. predict the INTERNAL_HOLDOUT target distribution of `n_{m+1}` as `T_{w_-s}(p_m)`;
4. calculate Jensen-Shannon distance to the actual INTERNAL_HOLDOUT `p_{m+1}`.

Identity baseline: predict `p_{m+1}` simply as `p_m`.

Per edge define

`DELTA = JS(identity,target) - JS(operator,target)`.

Positive DELTA means a shared operator predicts the next e-level better than mere contextual similarity.

For each skeleton, average DELTA across its evaluable edges. Primary statistic is the median skeleton-level DELTA.

## Primary null

10,000 deterministic skeleton-level sign-flip nulls.

For each null, independently reverse the orientation of all TRAIN ladder edges belonging to a selected skeleton with probability 0.5 before estimating the cross-skeleton operator. INTERNAL_HOLDOUT evaluation orientation remains the true `m -> m+1` direction.

This preserves:

- each skeleton's two context distributions;
- e-ladder similarity magnitude;
- frequencies and context marginals;
- within-skeleton dependence;

while destroying a consistent shared direction of change.

Report one-sided empirical p and z relative to null medians.

## Independent TRAIN-half transfer

Partition eligible TRAIN skeletons deterministically by SHA256 parity into `SKEL-A` and `SKEL-B`.

- estimate one operator from all A edges and evaluate on B TRAIN edges;
- estimate one operator from all B edges and evaluate on A TRAIN edges.

Use the same DELTA definition. Report median skeleton-level DELTA for A->B and B->A. These are directional replication checks, not additional optimisation.

## Repeated-application diagnostic

For skeletons containing `m`, `m+1`, and `m+2` with each type >=20 TRAIN and >=5 HOLDOUT occurrences, estimate `w_-s` as above and compare on HOLDOUT:

- identity prediction from `p_m` to `p_{m+2}`;
- one-step `T_w(p_m)`;
- repeated `T_w(T_w(p_m))`.

Report whether two applications outperform both identity and one application. This diagnostic cannot rescue the primary gate and has no minimum sample requirement.

## Synthetic method qualification

Before interpreting Voynich, run 6 known-answer `SHARED_OPERATOR` and 6 matched `IDIOSYNCRATIC_OPERATOR` synthetic replicates.

Per replicate:

- 40 skeletons;
- 34 context categories on each side;
- base context distributions drawn from deterministic Dirichlet(0.7);
- two e-levels per skeleton;
- counts per level drawn deterministically from integers 80..240;
- operator log-weights have SD 0.35 and are centred.

`SHARED_OPERATOR`: all skeletons use the same operator on each side.

`IDIOSYNCRATIC_OPERATOR`: each skeleton uses an independently drawn operator with the same marginal magnitude distribution.

Apply exactly the same leave-one-skeleton-out prediction and 1,000 skeleton sign-flip nulls.

The method qualifies only if:

- >=5/6 SHARED replicates have median DELTA > 0 and p <= 0.01;
- <=1/6 IDIOSYNCRATIC replicates have p <= 0.01.

If method qualification fails, Voynich results are descriptive only and cannot open another model.

## Frozen Gate V13

`V13_SHARED_E_OPERATOR_SUPPORTED` only if all are true:

1. synthetic method qualification passes;
2. at least 12 untouched-evaluable ladder edges spanning at least 8 distinct skeletons exist;
3. primary median skeleton DELTA > 0;
4. primary sign-flip empirical p <= 0.01 and z >= 2.5;
5. both independent TRAIN transfer directions A->B and B->A have median skeleton DELTA > 0.

Otherwise:

`V13_NO_SHARED_E_OPERATOR_EVIDENCE`.

## Interpretation

A pass would establish only that increasing e-count by one has a reproducible cross-skeleton effect on boundary-context distributions beyond simple e-ladder similarity. It would not identify the algebra of the hidden state transformation.

A pass permits a new synthetic V14 to infer the smallest operator family consistent with the observed transformation. It does not permit Voynich plaintext search.

A fail means the V11 e-ladder result should remain interpreted as morphological/contextual family resemblance, not evidence for one reusable repeated operator; no new nucleus transducer is justified from the present line.

## Stopping rules

- no post-result change to parser, context vocabulary, thresholds, operator form, smoothing, null, or gate;
- no exploratory variant may rescue a failed primary gate;
- negative result is committed;
- no GPU is authorised; one CPU job should run synthetic calibration and binding ciphertext test together;
- no plaintext output is generated.
