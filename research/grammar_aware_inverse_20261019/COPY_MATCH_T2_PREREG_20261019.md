# COPY-MATCH-T2 preregistration — 2026-10-19, before seeing T2 scores

## Prior evidence and nonclaims
The preceding LF-T1 failed against within-quire production-block controls: matched-null effect +0.001155 bits/token with null SD 0.002095, z=.55, p=.31. Q1B inverse identity recovery failed, and IDENT-GATE observational source/production twins remain indistinguishable. We must **not** claim that any heldout association separates semantic content from deliberate copied repertoire: these can be observationally equivalent.

T2 tests whether the prior long-range **anchor-to-late-category** scoring improvement persists against an *explicit cross-quire copying-template model with the same training data and storage capacity*, rather than only an on-page cache. A negative result stops this feature branch; a positive outcome is an unexplained predictive contrast, NOT decipherment. This is a pilot, not confirmatory source-model qualification.

## Data, folds, leakage and frozen representation
Use exactly `lf_transfer_export_20261019.sql`: one original read-only RF-derived stored token table, canonical Herbal-A running text, Currier A, Davis 1, quire A–G only. One whole quire held out at a time; train on six other quires. All folio sides and physical bifolia held together automatically. First 15 tokens of a held-out folio are observable prefix but **not scored**. Predict subsequent token categories with the previously frozen per-fold top-96 TRAIN-only surface types plus OTHER; never create target vocabulary using heldout data. Token positions preserve folio/paragraph ordering as exported.

M0/M1 exactly as `lf_transfer_score_20261019.js`: unconditional+context smoothed category probabilities, followed by within-folio online exact/one-edit copy kernel (decay .98), fixed mixtures and priors.

For all new alternatives, **identical training allocation**: each training folio has its early unique anchors (among 15 first tokens, length ≥4, frequency across training tokens 2–35, at most 10 chosen rarest) and its late count vector of 97 vocabulary categories. A donor is selectable only when it has at least one exactly shared early anchor. Source item labels never observed; image/morphology not used.

**Mcopy** (hard copy-template negative): select each training folio donor by weighted overlap of its anchor set with the test folio's early anchor set; inverse TRAIN frequency weight `1/freq(a)` per shared anchor. Aggregate *complete late donor block* histograms, normalize by each donor's late length to avoid length dominance; apply exactly the same one-edit neighbour kernel used by M1 to model scribal respellings. Give 35 pseudocounts towards training-global category probability. If there is no matching donor, use M1. Final `0.8 M1 + 0.2 Mcopy_template`.

**Massoc**: the original prior M2 anchor-to-late cooccurrence distribution, computing a Dirichlet-35-smoothed late-category distribution for each anchor from the same training folio/block counts; average over the available matching early anchors. To match copying capacity, apply the same fixed one-edit kernel to each training late category vector. Final `0.8 M1 + 0.2 Massoc_template`. No learned source embedding or added semantic labels. Same 97-dimensional training vectors, donors, 1-edit kernel, priors and 20% mixture. They differ only in how early anchors weight donor pages (template-conditioned vs anchor-conditional estimates).

**Hybrid diagnostic**: `0.8 Mcopy + 0.2 Massoc` over already normalized predictive distributions (M1 included in each); this explicitly adds predictor capacity and cannot be interpretively promoted, even if successful.

## Decision rule; matched block nulls
Primary signed contrast `D=mean_t log2(P_assoc(t)/P_copy(t))` heldout across all seven quires, plus per-quire and leave-one-quire-out descriptive bounding. Secondary copying improvement `mean_t log2(P_copy(t)/P_M1(t))`.

Hard null: 199 seeded permutations (xorshift32 seed 20261019) of *training folios' late blocks* among their four closest-length folios within each quire (leaves all held-out text, early anchors and reference M1 completely intact). Recompute BOTH copy and associative predictors each permutation; match same training memory and category distributions. Null tests whether the observed early-to-late association structure improves over a source-reuse-preserving block rearrangement. It is not a genuine cross-quire copy-template generative null: if D passes, identifiability is still not proven.

T2 pilot pass requires ALL: raw D ≥+0.02 bits/token; excess over null ≥2 null SD; permutation upper-tail p ≤.05; D>0 on ≥5/7 quires. Otherwise `FAIL/UNRESOLVED`. Do not threshold the secondary or the hybrid. Store full null distribution, per-fold scores, dataset counts, source-query and scorer code.

## Positive calibration, provenance and full grammar limitations
No claim of mechanistic identification is possible without a matched-volume/capacity synthetic source-vs-copying positive/negative assay with ≥20 independently generated instances, ≥90% sensitivity and ≤5% FPR. Both families must pass through the fully frozen SELECT/FORM/CONNECT/LINE_ENTRY renderer and identical physical-fold layout. Current pipeline M0/M1 approximates only observable form/context; it **does not run the full grammar renderer**. This T2 test is a stop/go audit for one feature, not a substitute for the missing calibration.

Original transcript provenance of `vms_directional_token_v01` is unverified: it is a stored derived table with no table comment, not an independently sourced TTLI/ZLZI paired panel. `voynich_semantic_notation_v1.rf_occurrences` also appears RF-derived and is NOT evidence of another transcription witness. Neither can qualify independent transcription robustness. Do not silently replace this requirement with same-source-table agreement. Re-run on authenticated independent layers only if available and keep per-layer readouts separate.

## Pre-score hazards audited
- No circular species-name anchor or translation/dictionary search.
- No heldout vocabulary, donor, weights or mixture optimisation.
- No scribe/Currier/section contamination; all are fixed by cohort restriction.
- No physical-unit leak: heldout unit is entire quire (stronger than bifolium).
- Primary model comparison uses exactly shared memory capacity and edit kernel; any apparent advantage is estimator-weighting rather than automatic source existence.
- No source/production identification if observationally equivalent.
