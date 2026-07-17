# AMENDMENT 006 — P2 foreground-token ViT + VLAD v1.6

**Date:** 2026-07-17  
**Status:** prospectively frozen before any P2 metric  
**P1 status:** terminally closed  
**Voynich Phase I opened:** no  
**Davis labels loaded:** no  
**f115r boundary loaded:** no

## Trigger

The full P1 SAGHOG v1.5.1 branch learned a statistically significant and perturbation-robust writer representation, but failed the frozen 1.5× acquisition-nuisance ratio gate and both synthetic-K gates. P1 is closed without terminal-test tuning.

P2 tests the distinct mechanism described by Raven, Matei and Fink, *Self-Supervised Vision Transformers for Writer Retrieval* (arXiv:2409.00751): retain local ViT patch tokens only when the corresponding image patch contains handwriting, and aggregate those local descriptors with VLAD rather than reducing a page to a class token or token moments.

No P2 metric has been observed before this amendment.

## Immutable backbone

- Hugging Face repository: `facebook/dinov2-small`.
- Immutable repository revision: `ed25f3a31f01632728cabb09d1542f84ab7b0056`.
- Weight file: `model.safetensors`.
- Expected weight SHA-256: `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`.
- Architecture: DINOv2 ViT-S/14.
- Hidden dimension: 384.
- Layers: 12.
- Patch size: 14.
- Parameters: approximately 22 million.
- Backbone is frozen; there is no writer-label or page-label fine-tuning.
- Features are the final normalized patch-token outputs. The class token is excluded.

This branch tests foreground selection and local-descriptor aggregation. It is not a revival of v1.3 DINOv3-7B whole-tile mean/standard-deviation pooling.

## Image representation and windowing

- Input source: Historical-WI colour pages already used by the external programme.
- Physical pages, not colour/binary derivatives, remain the unit of splitting and retrieval.
- Pages are converted to grayscale and binarized with Sauvola thresholding, window size 51.
- Model input uses a white background and black foreground replicated across three channels.
- Page windows: 224 × 224 pixels.
- Evaluation stride: 224 pixels, with right and bottom padding to the next complete window.
- Windows with foreground occupancy below 2.5% are discarded.
- No page-layout crop selection or keypoint sampling is used.

## Foreground-token rule

Each 224 × 224 window yields a 16 × 16 grid of ViT-S/14 patch tokens. A token is retained only when its corresponding 14 × 14 binary image patch contains at least 10 foreground pixels. The threshold is fixed before execution and is not selected on validation or terminal data.

All qualifying foreground tokens from all qualifying windows of a physical page form that page's local-descriptor set.

## VLAD aggregation

- Codebook fitting data: foreground tokens from training writers only.
- Codebook sampling cap: 1,000,000 training tokens, sampled deterministically and uniformly without replacement if exceeded.
- Clustering: MiniBatchKMeans.
- Centroids: 100.
- Initialization: k-means++.
- `n_init`: 3.
- Batch size: 8,192.
- Random seed: `20260719`.
- Each page token is assigned to its nearest centroid.
- Per-centroid residuals are summed and concatenated.
- Signed power normalization exponent: 0.5.
- L2 normalization follows power normalization.
- PCA with whitening is fitted on training-page VLAD vectors only.
- Formal output dimension: 384.
- Final L2 normalization follows PCA.

The exact resulting validation and test matrices must be persisted. Selection, retrieval, permutation, perturbation and synthetic-K tests must consume those exact matrices.

## Data split and selection

P2 uses a new deterministic writer-disjoint split of the complete 394-writer Historical-WI training corpus, seed `20260719`:

- 70% training writers;
- 15% validation writers;
- 15% terminal test writers.

All three physical pages of each writer remain in one partition. Training writers fit the VLAD codebook, PCA and nuisance regressions. Validation writers may choose only among the four prospectively fixed representations:

1. raw VLAD-PCA;
2. residualized against acquisition nuisance;
3. residualized against ink nuisance;
4. residualized against combined nuisance.

No backbone layer, foreground threshold, codebook size, PCA dimension or window stride is selected using validation or terminal results.

## Evaluation and gates

The same external gates remain in force:

- selected mAP exceeds acquisition-nuisance mAP by at least 0.05;
- selected mAP is at least 1.5 times acquisition-nuisance mAP;
- permutation p ≤ 0.01 with at least 199 permutations;
- each frozen perturbation retains at least 0.80 of unperturbed mAP;
- exact-K recovery at least 0.70;
- within-one-K recovery at least 0.90;
- synthetic null false-discrete rate at most 0.05;
- synthetic null abstention at least 0.95;
- at least 40 panels per formal mechanism.

Passing Historical-WI alone does not unlock Voynich. Independent external-family transfer and the remaining frozen calibration requirements must also pass.

## Preflight

Before a formal P2 execution, one miniature end-to-end preflight must verify:

- immutable checkpoint retrieval and SHA-256;
- finite DINOv2 inference;
- exact 16 × 16 patch-token alignment;
- foreground filtering;
- training-only VLAD codebook fitting;
- page-level VLAD encoding;
- training-only PCA/whitening;
- checkpoint and feature serialization;
- exact-matrix reuse by retrieval and a bounded permutation test;
- no Voynich, Davis or f115r access.

The preflight may reduce the PCA output dimension to the largest rank supported by its miniature training panel and may reduce permutation/panel counts. Those are smoke-only operational reductions and cannot alter the formal specification above.
