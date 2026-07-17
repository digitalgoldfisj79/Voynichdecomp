# AMENDMENT 006 — v1.6 continuous handwriting-style similarity

**Date:** 2026-07-17  
**Status:** prospectively frozen after the v1.5.1 external result and before any Voynich access or v1.6 metric  
**Voynich Phase I opened:** no  
**Davis labels loaded:** no  
**f115r boundary loaded:** no

## Trigger and decision

The audited SAGHOG v1.5.1 full run is Category B:

- selected terminal writer-retrieval mAP 0.4107341042252194;
- acquisition nuisance mAP 0.3053187055502651;
- absolute margin 0.10541539867495431: pass;
- writer/acquisition ratio 1.3452634796317764: fail against the frozen 1.5× writer-identification gate;
- permutation p=0.005 with 199 permutations: pass;
- all five perturbation retentions ≥0.80: pass;
- exact-K 0.13333333333333333 and within-one-K 0.2222222222222222: fail.

Therefore v1.5.1 does not validate writer identification or hand enumeration. It does justify a new, separately gated question:

> Does distance in a frozen handwriting representation provide a calibrated, nuisance-resistant ranking of relative writing-style similarity between different physical pages or fragments?

No v1.6 result may be used to retroactively declare the v1.5.1 writer-identification or K gates passed.

## Mandatory artifact-recovery stage

The v1.5.1 recovery bundle contains `result.json`, `writer_split.json` and `exact_features.npz`, but not:

- the selected `saghog_v15_best.pt` checkpoint;
- PCA model parameters;
- fold-local acquisition, ink and combined residual-model parameters;
- a complete reusable inference-pipeline state.

Consequently, the audited `resid_combined` representation cannot be applied to a new external corpus from the recovered bundle alone.

Before v1.6 external validation, run a documented v1.5.2 artifact-recovery replication with no scientific changes to architecture, data, splits, training schedule, augmentations, checkpoint-selection rule, PCA dimension, residualization or metrics. The only permitted changes are:

1. persist the selected checkpoint;
2. persist PCA and all residual-model parameters;
3. persist the immutable inference code and environment manifest;
4. include all required objects in a hash-verified bundle or durable repository;
5. correct the `result.json` self-hash ordering defect;
6. retain exact raw and selected train/validation/test matrices;
7. report reproducibility against v1.5.1.

The replication is acceptable for use in v1.6 only if:

- the same checkpoint-selection rule is used without hard-coding step 500;
- the selected representation remains `resid_combined`, or any divergence is treated as instability and investigated before proceeding;
- terminal selected mAP differs from 0.4107341042252194 by no more than 0.02 absolute;
- all v1.5.1 implemented gate decisions remain unchanged;
- all persisted hashes verify after job termination.

If these conditions fail, stop and classify P1 as unstable rather than selecting the more favourable run.

## Corpus admissibility

The primary v1.6 terminal corpus must be selected before inspecting model performance and must satisfy all feasible conditions below:

1. same-writer labels across at least two different physical pages or fragments;
2. different writers represented within a shared manuscript, collection or acquisition environment;
3. same-writer examples spanning heterogeneous acquisition conditions where available;
4. no writer identity recoverable from filename, page dimensions, background, derivative status or layout alone;
5. sufficient metadata to exclude colour/binary derivatives and adjacent crops from the same source image;
6. at least 40 eligible writers and at least three physical samples per writer for the primary terminal analysis;
7. legal and technically reproducible access with immutable source checksums.

HisFrag20 is a preferred candidate only if a metadata audit proves that it meets these conditions. Candidate choice must not be based on observed SAGHOG performance.

If no available corpus satisfies all conditions, v1.6 must report the unmet conditions and use the strongest available corpus as exploratory evidence only. It must not open Voynich.

## Frozen data separation

- Representation training remains external to Voynich.
- Corpus-level writer splits are disjoint.
- Pair-generation rules and nuisance strata are fixed from metadata before embedding evaluation.
- Any distance calibration is fitted on validation writers only.
- Thresholds, fragment evidence requirements and nuisance matching are selected on validation writers and applied once to terminal writers.
- No terminal writer or pair may influence model, residualization, calibration or hyperparameters.

## Primary pair task

Construct pairs only from different physical pages or source fragments.

### Positive pairs

Same writer, different physical page/fragment. Prefer different acquisition or manuscript strata where available.

### Negative pairs

Different writer, matched as closely as possible on manuscript/acquisition bucket, page dimensions, background statistics, ink density, foreground-patch count and coarse layout.

Each physical sample may occur in multiple pairs, but uncertainty must be clustered by writer and physical page. The raw number of pairs must never be treated as the independent sample size.

## Primary metrics

Report on the held-out terminal writers:

- ROC-AUC;
- average precision and pair prevalence;
- equal-error rate;
- Brier score;
- expected calibration error using ten fixed equal-frequency validation bins;
- same-writer and different-writer distance distributions;
- writer- and page-clustered 95% bootstrap confidence intervals with at least 2,000 resamples.

## Primary continuous-similarity gate

The representation passes as a continuous style-similarity instrument only if all conditions hold on the primary terminal corpus:

1. ROC-AUC lower clustered 95% confidence bound >0.65;
2. average precision exceeds pair prevalence by at least 0.15 absolute;
3. ROC-AUC exceeds the best nuisance-only control by at least 0.05 absolute;
4. the lower confidence bound for the model-minus-best-nuisance AUC difference is >0;
5. equal-error rate ≤0.35;
6. permutation p≤0.01 using at least 999 writer-block permutations;
7. all required cross-page and nuisance-matched sensitivity analyses below preserve at least 80% of the primary AUC excess above 0.5;
8. no negative control independently satisfies conditions 1–6.

Passing this gate validates relative similarity ranking only. It does not validate writer identity, a writer count, hard clustering or K=5.

## Required sensitivity analyses

### Cross-page retrieval

For every query, the gallery must exclude:

- the same physical page;
- colour/binary or resolution derivatives;
- adjacent crops from the same source image.

Report mAP, top-1 and top-5 with writer-clustered confidence intervals.

### Acquisition and layout contrasts

Report separately:

- same writer / different acquisition;
- same writer / different layout;
- different writer / same acquisition;
- different writer / same manuscript or layout stratum.

A model that succeeds only when writer and acquisition are aligned fails v1.6.

### Fragment-length calibration

Evaluate, without replacement where possible:

- whole physical page or full supplied fragment;
- one line;
- word-like crop;
- fixed foreground-patch budgets of 8, 16, 32, 64 and 128.

For each evidence level, report AUC, AP, EER, confidence interval and repeated-subsample stability. Select the minimum admissible evidence on validation writers only. Terminal fragments below that evidence level require abstention.

### Content conditioning

Where transcription or grapheme labels exist, compare pairs matched on content and pairs deliberately mismatched on content. Where labels do not exist, use frozen visual-content strata derived without writer labels. Shared textual content must not explain the full style signal.

### Perturbation robustness

Evaluate contrast, brightness/background replacement, translation, scale, erosion, dilation and synthetic scan degradation. Each perturbation must retain at least 80% of the unperturbed AUC excess above 0.5.

## Negative controls

Run all of the following through the identical pair and bootstrap machinery:

- background-only features;
- ink-mask-only features;
- acquisition-only nuisance vectors;
- combined acquisition + ink nuisance vectors;
- shuffled foreground patches within page;
- spatially scrambled strokes;
- writer labels permuted in writer blocks.

A negative control AUC above 0.60 or within 0.05 of the full model triggers failure and diagnosis, not post-hoc control removal.

## Exploratory outputs kept separate from the primary gate

- continuous neighbourhood graphs;
- local boundary/discontinuity scores;
- manifold visualisations;
- hierarchical clustering;
- continuous-versus-discrete model comparison.

These may diagnose structure but cannot rescue a failed primary gate.

## Frozen interpretation ladder

- **v1.6 pass:** continuous relative style similarity is externally validated; proceed to a sealed blind Voynich Phase I that returns only continuous matrices, stability, discontinuity and abstention outputs.
- **Significant but nuisance-sensitive:** do not open Voynich; seek a better cross-acquisition corpus or move to P2/classical/HTR fusion.
- **Non-significant or unstable:** close SAGHOG P1 for this use case and move to P2/classical/HTR alternatives.
- **K failure:** remains irrelevant to the continuous claim and prohibits writer-count statements.

## Voynich Phase I, only after a v1.6 pass

The first sealed Voynich run must return, before any Davis or f115r reveal:

- all-pairs folio similarity matrix;
- neighbourhood stability under patch and quire bootstrap;
- local page and line discontinuity scores;
- quire-aware uncertainty;
- continuous-versus-discrete model comparison;
- evidence-based abstention.

No forced K=5. Davis hand labels, Davis assignments, f115r boundary, section labels, Currier labels and revealing folio identities remain sealed until these outputs and hashes are frozen.