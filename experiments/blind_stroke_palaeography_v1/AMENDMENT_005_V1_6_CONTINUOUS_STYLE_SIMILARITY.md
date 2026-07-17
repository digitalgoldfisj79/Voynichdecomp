# AMENDMENT 005 — Continuous handwriting-style similarity v1.6

**Date:** 2026-07-17  
**Status:** prospectively frozen after the v1.5.1 full-run audit and before any v1.6 metric  
**Voynich Phase I opened:** no  
**Davis labels loaded:** no  
**f115r boundary loaded:** no

## Trigger

The completed SAGHOG v1.5.1 external run (`6a5a1540d216bd6f3a1fb177`) produced statistically significant and perturbation-robust writer retrieval, but it failed the frozen 1.5× acquisition-nuisance ratio and discrete-K recovery gates.

The result is therefore category B:

> writer retrieval is significant and robust, but nuisance-ratio and K-recovery gates fail.

This is not a writer-identification pass, does not recover the number of hands, and does not justify opening Voynich. It is sufficiently positive to test a narrower capability prospectively: calibrated relative handwriting-style similarity.

The v1.5 and v1.5.1 writer-identification and K gates remain unchanged. This amendment does not weaken, reinterpret or replace them.

## Frozen scientific target

The v1.6 primary question is:

> Does distance in the frozen SAGHOG representation provide a calibrated, nuisance-resistant ranking of handwriting-style similarity between samples from different physical pages?

A v1.6 pass may support relative statements such as:

- sample A is more stylistically similar to sample B than to most comparison samples;
- a proposed page or line boundary shows a continuous style discontinuity;
- a page is an outlier relative to neighbouring pages;
- style variation is better represented as continuous than as discretely clustered.

A v1.6 pass may not support a categorical writer count or a claim that there are five Voynich scribes. No forced K=5 is permitted.

## Frozen v1.5.1 source representation

The source model and representation are fixed as follows:

- external job: `6a5a1540d216bd6f3a1fb177`;
- launcher commit: `f376ee2a560dbbd1a0d2a3f06402cc70ec48b556`;
- assembled source SHA-256: `fd8f93893a488b59d41eba4395de82e5690ebb491bc8bbe6c1de581a2884cdd8`;
- upstream SAGHOG commit: `123cf0f306f105a46edbe8def06f49b54e64832e`;
- selected checkpoint: metric-learning step 500;
- selected checkpoint SHA-256: `9b543dbc13600a8a32ca04e7794541d8e184c0cbcd793e1b9492d37e03445d09`;
- selected representation: `resid_combined`;
- output dimension: 512;
- exact external feature matrix SHA-256: `dd47d63b635ea2f4722920221b146491b59f82b63f5b2fbffe6c12c6c06f9a52`.

Cosine similarity in the frozen 512-dimensional `resid_combined` representation is the primary similarity score. Any probability calibration is fitted on calibration writers only.

## Checkpoint-persistence prerequisite

The v1.5.1 recovery bundle contains `result.json`, `writer_split.json` and `exact_features.npz`, but not `saghog_v15_best.pt`. The completed job reports the checkpoint size and SHA-256, but its bytes were not persisted in the bundle.

No new-corpus v1.6 inference may begin until one of the following is completed and documented:

1. recover the exact checkpoint bytes and verify SHA-256 `9b543dbc13600a8a32ca04e7794541d8e184c0cbcd793e1b9492d37e03445d09`; or
2. if exact recovery is impossible, perform one documented deterministic reproduction run for the sole purpose of persisting the checkpoint and all model-level artifacts.

A reproduction is not presumed identical. Its source, package lock, hardware, random-state controls, checkpoint hash and all v1.5.1 metrics must be reported. It is accepted as a replacement only if the predeclared reproducibility tolerances below are met on the identical external split:

- selected checkpoint remains step 500;
- selected representation remains `resid_combined`;
- validation and test mAP for every representation differ by no more than 0.01 absolute;
- nuisance mAP values differ by no more than 0.01 absolute;
- every gate decision is unchanged;
- all feature arrays are finite and writer splits are byte-identical.

The environment must pin all scientific dependencies, including `scikit-learn`, and explicitly set KMeans `n_init`; the v1.5.1 launcher left this default version-sensitive.

## External-corpus suitability gate

Before model inference, a metadata-only dataset audit must establish that the terminal corpus can test style rather than acquisition identity. The corpus must provide:

- writer labels and physical-page identifiers;
- at least two different physical pages for each evaluated writer;
- same-writer samples spanning heterogeneous acquisition, source, session or manuscript conditions where possible;
- different writers sharing acquisition or manuscript conditions;
- enough metadata to match or condition on page dimensions, background, ink density and layout;
- exclusion of colour, binary or other derivatives of the same physical image from cross-page positives;
- exclusion of adjacent crops from the same source image from query–gallery comparisons.

HisFrag20 remains a candidate only if its metadata satisfies these conditions. Dataset acceptance must be frozen before embeddings are computed.

## Split and leakage controls

Writers are divided deterministically into training, calibration and terminal-test partitions. No writer, physical page or image derivative may cross partitions.

Training writers may fit nuisance residualization, score calibration and any fixed preprocessing parameters. Calibration writers may select only predeclared calibration choices. Terminal writers are evaluated once.

Every positive query–gallery pair must use different physical pages. Where a writer appears in more than one manuscript or acquisition source, the primary positive pair must cross that boundary. Same-image and same-page pairs are prohibited from primary endpoints.

## Primary endpoints

The primary unit is a pair of samples from different physical pages, represented by a fixed draw of 96 foreground patches per sample.

The following are reported on terminal writers with 95% confidence intervals clustered by writer:

1. same-writer versus different-writer ROC-AUC;
2. average precision and improvement over the terminal positive-pair prevalence;
3. equal-error rate;
4. Brier score and expected calibration error from a calibration-writer-only mapping;
5. cross-page retrieval mAP, top-1 and top-5;
6. Spearman rank stability of pairwise similarities under perturbation.

All pair-sampling seeds and pair manifests are persisted.

## Matched nuisance contrasts

The terminal test must contain both of the following contrasts:

1. same writer, different acquisition/layout;
2. different writer, matched acquisition/layout.

Matching variables include, where available:

- acquisition source or manuscript;
- page dimensions and aspect ratio;
- mean background colour and contrast;
- ink density and darkness;
- coarse line count, column structure and layout.

The frozen SAGHOG score is compared with acquisition-only, ink-only and combined-nuisance scores on the identical pair manifest.

## Content conditioning

Where external transcription or grapheme labels exist, the analysis must stratify or match pairs by textual or glyph content. At minimum it reports:

- matched-content same-writer versus different-writer discrimination;
- mismatched-content same-writer discrimination;
- the residual writer effect after conditioning on content features.

If content labels are unavailable, this limitation is explicit and the corpus cannot alone establish glyph-conditioned style similarity.

## Fragment-length calibration

The fixed evidence ladder is:

- whole-page foreground pool;
- 128 foreground patches;
- 96 foreground patches;
- 64 foreground patches;
- 32 foreground patches;
- 16 foreground patches;
- one line where line labels exist;
- one word-like crop where word labels exist.

For each level, the analysis reports discrimination, retrieval, calibration, bootstrap uncertainty and repeated-subsample stability. The minimum usable evidence level is the smallest level whose median pairwise similarity has test–retest Spearman correlation at least 0.80 and whose ROC-AUC is within 0.05 of the 96-patch primary endpoint.

## Perturbation robustness

Each terminal sample is evaluated under independently generated:

- contrast change;
- brightness/background replacement;
- translation;
- scale change;
- erosion;
- dilation;
- synthetic scan degradation.

For every perturbation, report retrieval retention, ROC-AUC retention and Spearman rank correlation against the unperturbed all-pairs score matrix.

## Negative controls

The terminal analysis includes:

- shuffled foreground patches;
- background-only input;
- ink-mask-only input;
- spatially scrambled strokes;
- acquisition-only nuisance vectors;
- ink-only nuisance vectors;
- combined-nuisance vectors;
- writer-label permutation.

Negative controls consume the same pair and bootstrap manifests as the primary representation.

## Frozen v1.6 pass criteria

The representation is accepted as a continuous page-style similarity instrument only if all primary criteria pass:

1. terminal cross-page ROC-AUC is at least 0.70 and its writer-clustered 95% lower confidence bound exceeds 0.60;
2. average-precision lift over pair prevalence is at least 0.10 and its 95% lower confidence bound exceeds zero;
3. ROC-AUC exceeds each acquisition, ink and combined-nuisance baseline by at least 0.05, with a paired writer-clustered 95% lower confidence bound above zero;
4. the same-writer/different-acquisition and different-writer/matched-acquisition contrasts each have ROC-AUC at least 0.65 and a 95% lower confidence bound above 0.55;
5. every named perturbation has retrieval and ROC-AUC retention at least 0.80 and rank correlation at least 0.80;
6. calibrated expected calibration error is at most 0.10 and Brier score improves on the prevalence-only predictor;
7. each non-permutation image negative control has ROC-AUC at most 0.55 and is at least 0.10 below the primary representation;
8. the writer-label permutation test has p ≤ 0.01 with at least 999 permutations;
9. the result replicates on a second external terminal corpus or on a predeclared cross-manuscript/acquisition holdout within the primary corpus.

Failure of any criterion means v1.6 does not validate direct Voynich use. Results may be diagnostic but cannot be used to open the seal.

## Multiple comparisons and uncertainty

The primary endpoints and gates above are fixed. Secondary fragment, perturbation and content analyses report Holm-adjusted p-values where multiple null hypotheses are tested. Confidence intervals use at least 2,000 writer-cluster bootstrap replicates. Pair-level resampling without writer clustering is prohibited as the primary uncertainty estimate.

## Decision rule after v1.6

- **Pass:** the representation may proceed to a blind Voynich continuous-similarity phase. It remains ineligible for writer-count claims.
- **Fails only on cross-acquisition or nuisance criteria:** classify as acquisition-confounded and do not apply directly to Voynich.
- **Non-significant, unstable or negative-control failure:** close SAGHOG P1 for this use case and move to P2, classical skeleton/contour or historical-HTR alternatives.

## Blind Voynich phase after a pass

Only after all v1.6 outputs and decisions are frozen may the model process blind Voynich samples. The first sealed output must contain:

- all-pairs folio similarity matrix;
- neighbourhood stability;
- local discontinuity scores;
- quire-aware bootstrap uncertainty;
- continuous-versus-discrete model comparison;
- evidence-length and abstention maps.

Folio identities may be retained only as opaque internal identifiers. Davis hand labels, Davis five-hand assignments, section labels, Currier labels, f115r boundary information and forced K=5 remain unavailable until the blind outputs are frozen.
