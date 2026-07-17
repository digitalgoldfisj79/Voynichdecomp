# AMENDMENT 005 — SAGHOG v1.5 full external development and confirmation

**Date:** 2026-07-17  
**Status:** prospectively frozen before any v1.5 metric  
**Voynich Phase I opened:** no  
**Davis labels loaded:** no  
**f115r boundary loaded:** no

## Trigger

The v1.4 miniature Historical-WI smoke completed end to end and produced substantial writer retrieval signal, but it was not faithful enough to the declared full P1 implementation to justify a formal launch. In particular, the smoke omitted morphology augmentations, used a smoke-only 64-dimensional PCA, and ran only 160 HOG-MAE and 120 metric-learning updates. Its metric loss was essentially flat and its K selector failed. Version 1.4 is closed as an infrastructure and feasibility smoke.

## Immutable implementation basis

- SAGHOG upstream repository: `marco-peer/icdar24`
- upstream commit: `123cf0f306f105a46edbe8def06f49b54e64832e`
- immutable v1.4 helper source: commit `a92b5899b1e9707817058b00b9a1abd259250a5f`
- helper path: `experiments/blind_stroke_palaeography_v1/code/saghog_v1_4_smoke.py`
- helper bytes: `19040`
- helper SHA-256: `55a6aac2f6fa831e6624c57b57ade5d49d8994ebc8f420b4f267a56c68dabeeb`

v1.5 imports only data parsing, patch extraction, nuisance construction, retrieval, residualization and public-model loading helpers from that immutable source. The v1.5 training, checkpoint selection, perturbation, permutation and K-calibration logic is separately frozen.

## Development corpus and split

Historical-WI colour pages are the development corpus.

- all eligible writers with exactly three physical pages are admitted;
- deterministic writer split seed: `20260718`;
- 70% training writers, 15% validation writers, remainder development-test writers;
- all pages of a writer remain in one partition;
- 128 deterministic SIFT/foreground patches per page where available;
- no colour/binary duplicate is admitted;
- validation writers select checkpoint and representation;
- development-test writers are evaluated once after selection.

Because v1.4 aggregate Historical-WI results have already been observed, the v1.5 Historical-WI development-test is not the terminal cross-corpus confirmation. HisFrag20 remains sealed for that purpose.

## P1 architecture

- 32 × 32 local handwriting patches;
- MAE patch size 4;
- encoder depth 8;
- embedding dimension 512;
- HOG pool 4;
- HOG bins 9;
- mask ratio 0.75;
- decoder depth 1;
- NetRVLAD with 100 clusters;
- public multi-similarity loss: alpha 2, beta 40, base 0.2;
- public multi-similarity miner epsilon 0.1;
- pseudo-labels: deterministic MiniBatchKMeans on training-writer SIFT descriptors, 128 clusters;
- PCA output dimension 512 for the full development run.

## Frozen augmentation policy

Training inputs are transformed prospectively as follows:

- random grayscale with probability 0.20;
- random foreground/binarized view with probability 0.20;
- random erosion with probability 0.30;
- random dilation with probability 0.30;
- erosion and dilation use deterministic seeded random 3 × 3 binary kernels with centre set to one;
- evaluation receives no stochastic augmentation and uses grayscale three-channel inputs normalized to the public colour-evaluation convention.

The HOG target is always generated from the unaugmented foreground mask corresponding to the source patch.

## Frozen training schedule

### HOG-MAE pretraining

- optimizer: AdamW;
- learning rate: `8e-4`;
- weight decay: `0.05`;
- batch size: `512` patches;
- updates: `5,000`;
- gradient clipping: `0.02`;
- deterministic seed: `20260718`.

### NetRVLAD metric fine-tuning

- optimizer: AdamW;
- learning rate: `1e-3`;
- weight decay: `0.01`;
- batch construction: 32 pseudo-classes × 16 patches = 512 patches;
- updates: `5,000`;
- gradient clipping: `1.0`;
- validation checkpoint evaluations at updates 500, 1,000, ..., 5,000;
- checkpoint selected solely by validation-writer raw retrieval mAP;
- ties resolved in favour of the earlier update.

The terminal development-test partition is not evaluated during training or checkpoint selection.

## Representation selection

The selected checkpoint yields page representations by power-normalized sum aggregation of local NetRVLAD descriptors, followed by training-only PCA whitening to 512 dimensions.

Validation chooses one of:

- `raw`;
- `resid_acquisition`;
- `resid_ink`;
- `resid_combined`.

Residual models are fitted on training pages only. The selected representation name is applied to the development-test partition without modification. Exact matrices are persisted and reused for every downstream statistic.

## Development evaluation

The development-test evaluation includes:

- retrieval mAP, top-1 and top-5;
- acquisition, ink and combined nuisance baselines;
- 199 writer-label permutations on the exact selected matrix;
- synthetic K=2–10 calibration with at least 40 panels;
- perturbation retention under contrast, scale, erosion, dilation and translation;
- raw and residualized ablations;
- checkpoint and feature-matrix hashes.

Development criteria required before opening HisFrag20:

- selected mAP exceeds acquisition nuisance by at least 0.05;
- selected/acquisition mAP ratio at least 1.5;
- permutation p ≤ 0.01;
- perturbation retention at least 0.80 for every frozen perturbation;
- synthetic-K exact rate at least 0.70;
- synthetic-K within-one rate at least 0.90.

Failure of any criterion keeps HisFrag20 sealed and closes P1 as presently specified.

## Independent cross-corpus confirmation

Only after all Historical-WI development criteria pass may HisFrag20 be opened. No HisFrag20 item may influence pretraining, pseudo-label construction, NetRVLAD centres, PCA, checkpoint selection, residualization or thresholds.

HisFrag20 confirmation must reproduce the retrieval, nuisance, permutation, perturbation and K gates. Voynich remains sealed until both external stages pass.

## Exclusions

- no DINOv3 primary selection;
- no Davis labels;
- no Voynich crop, folio, section, Currier label or f115r boundary;
- no forced K=5;
- no scientific inference from infrastructure logs alone.
