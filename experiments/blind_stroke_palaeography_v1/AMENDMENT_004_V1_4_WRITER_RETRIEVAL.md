# AMENDMENT 004 — Handwriting-specific writer retrieval v1.4

**Date:** 2026-07-17  
**Status:** prospectively frozen before any v1.4 metric  
**Voynich Phase I opened:** no  
**Davis labels loaded:** no  
**f115r boundary loaded:** no

## Trigger

The v1.3 Historical-WI smoke completed technically but showed that the frozen general-purpose DINOv3 representation, reduced by simple token moment pooling, did not isolate writer signal from acquisition/layout nuisance. A post-run audit also found that permutation and synthetic-K tests did not consume the exact selected out-of-fold representation. Version 1.3 is closed.

## Model hierarchy

### Primary branch P1 — public SAGHOG reproduction

P1 implements the public reference code for:

Marco Peer, Florian Kleber and Robert Sablatnig, *SAGHOG: Self-Supervised Autoencoder for Generating HOG Features for Writer Retrieval*, ICDAR 2024.

Immutable upstream source:

- repository: `marco-peer/icdar24`
- commit: `123cf0f306f105a46edbe8def06f49b54e64832e`
- architecture: masked autoencoder predicting HOG targets, followed by NetRVLAD writer-retrieval fine-tuning

The public repository does not contain distributable pretrained checkpoints; its model-zoo entries point to the authors' private filesystem. P1 therefore trains from scratch on external handwriting only. It is a public-code reproduction, not a claim to possess the authors' released weights or their 24k-document SAM-preprocessed corpus.

Frozen P1 architecture and training defaults follow the public configuration unless explicitly stated below:

- input patches: 32 × 32 RGB converted to the frozen binarized/foreground view;
- patch size: 4;
- encoder depth: 8;
- embedding dimension: 512;
- HOG pool: 4; HOG bins: 9;
- masked-patch ratio: 0.75;
- decoder depth: 1;
- optimizer: AdamW;
- pretraining objective: masked HOG reconstruction;
- retrieval head: NetRVLAD, 100 clusters, 512-dimensional PCA/output representation;
- writer metric-learning objective: multi-similarity loss with the public margins and coefficients;
- morphological erosion/dilation augmentations retained.

### Primary branch P2 — foreground-token ViT + VLAD

P2 is an independent reproduction of the writer-retrieval design reported by Raven, Matei and Fink, ICDAR 2024: local foreground tokens from a self-supervised ViT aggregated with VLAD. P2 exists because foreground-token aggregation is directly testable and isolates whether P1's HOG reconstruction stage, rather than writer-specific aggregation, is the limiting element.

P2 may use an accessible ungated self-supervised ViT checkpoint, but the checkpoint identifier, digest, token layer, foreground rule, VLAD cluster count and dimensionality must be frozen in a subsequent implementation record before any P2 metric. P2 is not DINOv3-7B and will not use whole-page mean/std pooling.

### Supporting branches

- classical skeleton/contour/width features;
- historical HTR encoder features;
- DINOv3 only as an auxiliary dense-correspondence feature, never the primary selector.

## External data split and leakage controls

Historical-WI is split deterministically by writer using seed `20260717`:

- 70% training writers;
- 15% validation writers;
- 15% terminal test writers.

All pages and derivatives belonging to a writer remain in one partition. Colour and binarized derivatives of the same physical page share one page identifier. Self-supervised P1 pretraining and retrieval fine-tuning may use training writers only. Validation writers select epoch/checkpoint and fixed aggregation hyperparameters. Test writers are evaluated once after selection.

HisFrag20 is reserved as a second external corpus. No HisFrag20 test item may be used for pretraining, centre initialization, PCA, threshold selection or stopping.

## Nuisance controls

The previous whole-item nuisance vector was dominated by source/layout information. v1.4 reports three prospectively separate baselines:

1. `nuisance_acquisition`: colour/background, dimensions and coarse layout;
2. `nuisance_ink`: ink fraction, component counts and width summaries;
3. `nuisance_combined`: concatenation of 1 and 2.

Primary writer features are evaluated both raw and after fold-local residualization against each nuisance set. A corpus is rejected as a primary calibration corpus if writer retrieval is attributable mainly to acquisition/source features and the handwriting-specific branch cannot exceed the acquisition baseline under writer-disjoint held-out evaluation.

## Evaluation correction

Every candidate must return and persist its exact fold-local out-of-fold feature matrix. Model selection, retrieval mAP, permutation testing, perturbation testing and synthetic-K calibration must consume the identical persisted matrix. Mapping a selected representation name back to an unresidualized base matrix is prohibited.

## Gates before full calibration

The first execution is an infrastructure and miniature end-to-end smoke, not confirmatory evidence. It must demonstrate:

- immutable upstream source retrieval;
- dependency-complete import;
- finite HOG-MAE forward/backward pass;
- finite NetRVLAD forward/backward pass;
- deterministic writer-disjoint data manifest;
- checkpoint save/reload;
- exact OOF matrix persistence and reuse;
- one complete miniature pretrain → fine-tune → retrieval cycle.

Only after that smoke succeeds may a full P1 external training run be launched. No Voynich crop, label, folio or boundary is available to model selection.