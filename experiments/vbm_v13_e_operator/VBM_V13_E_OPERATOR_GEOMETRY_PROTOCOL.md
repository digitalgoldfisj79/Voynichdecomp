# VBM v13 — e-operator geometry programme

Date: 2026-09-02
Status: **PREREGISTERED BEFORE V13 OUTPUT**
Parent: V12 closeout `V12_COMPOSITIONAL_TRANSDUCER_FAILS_PRESSURE_TEST`.

## Purpose

V11 established that nuclei sharing an e-skeleton but differing in e multiplicity occupy unusually similar boundary contexts. V12 showed that a synthetic transducer with a global repeated e-operator is largely identifiable, but its Stage-B failures were entirely on the nucleus/e-operator side while the half-factorised bridge map recovered perfectly.

V13 therefore asks a narrower ciphertext-only question:

> Does adding one `e` behave like a shared repeated transformation across nucleus families, or is the V11 e-ladder effect only local family resemblance without a common operator?

No plaintext, language model, or Voynich decoding is permitted.

## Corpus and parser

Use the unchanged V11/Q0b parser and firewalls:

- ZLZI transcription from `voynich_transcriptions_slim.json`;
- atoms `ckh, cth, cph, cfh, ch, sh, qo` as left halves;
- final glyph as right half;
- token interior as nucleus;
- single-glyph tokens have empty nucleus;
- invalid tokens terminate a segment;
- H1/C1 folios remain excluded;
- SHA256 TRAIN/INTERNAL_HOLDOUT split remains identical to V11.

No parser change is allowed after output begins.

## e-ladders

For non-empty nucleus `n`:

- `skeleton(n)` replaces every maximal `e+` run by `E`;
- `m(n)` is total e-count.

Primary ladder steps are ordered pairs `(a,b)` such that:

- same skeleton;
- `m(b)=m(a)+1`;
- both types occur at least 20 times in TRAIN;
- both occur at least 5 times in INTERNAL_HOLDOUT.

If multiple types occupy the same skeleton/e-count level, every type is retained for contextual clustering but the primary step set uses deterministic frequency-ranked one-to-one matching within adjacent levels: highest-frequency source to highest-frequency target, second to second, etc., truncated to the smaller level cardinality. This prevents cross-product inflation.

The inferential unit for resampling is the ladder step/skeleton, never an occurrence.

## Context representation

V11-C showed that visible bridge halves contain reusable structure, so V13 does not treat a full bridge as atomic.

For each nucleus occurrence record four categorical boundary slots:

1. previous bridge right half (`prevR`) or EDGE;
2. previous bridge left half (`prevL`) or EDGE;
3. next bridge right half (`nextR`) or EDGE;
4. next bridge left half (`nextL`) or EDGE.

For each slot use the 32 most frequent TRAIN half-types plus OTHER and EDGE. Counts receive additive smoothing 0.5, are normalised per slot, square-root transformed (Hellinger embedding), and concatenated.

TRAIN vectors determine centroids. HOLDOUT vectors use the same slot vocabulary and are assigned to TRAIN centroids by Euclidean distance.

## Branch A — shared discrete permutation operator

### Hypothesis

There exists a low-cardinality latent contextual state system in which one additional `e` applies approximately the same permutation to every nucleus family.

### Candidate K

Predeclared grid:

`K = 2,3,4,5,6,7,8,9,10`

For each K:

1. cluster eligible TRAIN nucleus context vectors with deterministic KMeans (`n_init=64`, fixed seed);
2. form a KxK count matrix from TRAIN ladder-step source/target cluster labels;
3. use Hungarian assignment to obtain the single permutation `P_K` maximising explained TRAIN steps;
4. compute normalised TRAIN accuracy
   `NACC = (ACC - 1/K)/(1 - 1/K)`.

Choose `K*` by maximum TRAIN NACC; ties choose the smaller K. K selection never uses HOLDOUT.

Primary HOLDOUT statistic: assign HOLDOUT context vectors to the frozen TRAIN centroids and measure the fraction of the same ladder steps satisfying

`cluster_hold(target) = P_K*(cluster_hold(source))`.

Report raw accuracy and normalised accuracy.

### Familywise null

10,000 deterministic matched target-shuffle nulls. Within each e-increment level `m -> m+1` and target TRAIN-frequency tertile, shuffle target nucleus identities among ladder steps. For every null repeat the entire K-selection procedure on TRAIN and evaluate the selected K/permutation on HOLDOUT. This controls the K search and preserves source/target context marginals, e-level transition, and approximate frequency.

### Gate A

`A_SHARED_E_PERMUTATION_SUPPORTED` only if:

- at least 15 primary ladder steps exist;
- observed HOLDOUT normalised accuracy >= 0.50;
- observed HOLDOUT raw accuracy exceeds the 99th percentile of the familywise null;
- empirical p <= 0.01.

Otherwise `A_NO_SHARED_E_PERMUTATION`.

## Branch B — repeated-step composition

### Hypothesis

If one added `e` applies a shared operator P, two added e's should approximately apply `P^2`.

Use K* and P from Branch A without refitting.

Eligible two-step chains have same skeleton and observed types at e-counts `m`, `m+1`, and `m+2`, each meeting the same TRAIN/HOLDOUT frequency thresholds. Multiple types within a level are frequency-ranked one-to-one as above.

Primary statistic on HOLDOUT:

`cluster_hold(n_{m+2}) = P^2(cluster_hold(n_m))`.

10,000 target-shuffle nulls preserve m->m+2 level and target-frequency tertile. Branch B is inferential only if at least 5 two-step chains exist.

### Gate B

`B_ITERATED_E_OPERATOR_SUPPORTED` only if:

- Branch A passes;
- at least 5 two-step chains exist;
- raw HOLDOUT P^2 accuracy exceeds the 99th percentile null;
- empirical p <= 0.01;
- raw accuracy >= 0.60.

If fewer than 5 chains: `B_UNDERPOWERED_TWO_STEP`.
Otherwise failure: `B_NO_ITERATED_E_OPERATOR`.

## Synthetic method qualification

Before Voynich gates receive evidential interpretation, the same A/B pipeline must be calibrated on frozen V12 synthetic data generated by `vbm_v12_compositional_runner.py`.

Binding calibration sets:

- six Stage-A POS replicates (true shared global pi);
- six Stage-A NUC_BROKEN matched replicates (no shared nucleus operator).

The context representation is made structurally equivalent: synthetic bridge surface pair index is decomposed into its known `(R,L)` halves, and nucleus surface index into known `(s,m)` morphology. FIT/HOLD is 80/20 exactly as V12.

Method qualification requires:

- at least 5/6 POS replicates have Branch-A HOLD normalised accuracy >=0.50;
- POS median Branch-A HOLD normalised accuracy exceeds NUC_BROKEN median by >=0.25;
- at least 5/6 POS raw Branch-A HOLD accuracies exceed their matched NUC_BROKEN raw accuracies.

Branch B synthetic results are reported but do not veto Branch A qualification because two-step power depends on realised chains.

If synthetic qualification fails, Voynich A/B outputs are descriptive only and final verdict is `V13_METHOD_NOT_QUALIFIED`.

## Programme decisions

If method qualifies:

- A fail -> `V13_E_LADDER_SIMILARITY_WITHOUT_GLOBAL_OPERATOR`.
- A pass, B underpowered/fail -> `V13_ONE_STEP_OPERATOR_SIGNAL_ONLY`.
- A and B pass -> `V13_SHARED_ITERATED_E_OPERATOR_SUPPORTED`.

Only the final verdict permits a fresh synthetic V14 model using the empirically selected K* and the observed operator cycle structure. It still does not permit plaintext search.

## Evidence rules

- No plaintext or language score.
- No post-result K expansion or threshold change.
- No alternate parser or e-skeleton definition can rescue a failed gate.
- Nulls repeat K selection.
- Negative results are committed.
- No GPU is authorised; CPU execution only.
