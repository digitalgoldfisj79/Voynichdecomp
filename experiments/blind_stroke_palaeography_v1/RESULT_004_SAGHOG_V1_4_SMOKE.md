# SAGHOG v1.4 end-to-end smoke result

**Date:** 2026-07-17  
**Job:** `6a5a0d4cd216bd6f3a1fb006`  
**Status:** execution completed; promising writer signal; v1.4 remains smoke-only  
**Voynich Phase I opened:** no  
**Davis labels loaded:** no  
**f115r boundary loaded:** no

## Execution

The complete miniature writer-disjoint Historical-WI route executed successfully on an A100 using the immutable public SAGHOG source at commit `123cf0f306f105a46edbe8def06f49b54e64832e`.

- deterministic split: 24 train writers, 8 validation writers, 8 terminal test writers;
- three physical pages per writer;
- 64 local handwriting patches per page;
- 7,680 patches and 120 pages total;
- masked-HOG autoencoder training completed;
- NetRVLAD multi-similarity fine-tuning completed;
- validation-only representation selection completed;
- terminal test evaluated once;
- exact selected test matrix persisted and reused for retrieval, permutation and K smoke;
- checkpoint, split manifest and feature matrices were hashed;
- result bundle SHA-256: `3d6cce4b91886b1b2edbe410c163ee7165bc7e6761b263b8619137a364165857`.

## Smoke results

Validation selected `resid_combined`.

Terminal test metrics:

- raw SAGHOG mAP: `0.6315330469`; top-1 `0.6666666667`; top-5 `0.875`;
- selected `resid_combined` mAP: `0.5727761992`; top-1 `0.5416666667`; top-5 `0.875`;
- acquisition nuisance mAP: `0.3930494589`;
- ink nuisance mAP: `0.5322065836`;
- combined nuisance mAP: `0.5555284993`.

The selected representation exceeded acquisition nuisance by `0.1797267402` mAP, a ratio of `1.4572624034`. It exceeded combined nuisance by only `0.0172476999`, a ratio of `1.0310473718`. Raw SAGHOG exceeded combined nuisance by `0.0760045476`, but raw was not the validation-selected representation.

Permutation smoke: `p = 0.05` with only 19 permutations, the smallest attainable p-value under that smoke design. This is suggestive, not confirmatory.

Synthetic-K smoke:

- exact rate: `0.1428571429`;
- within-one rate: `0.3571428571`.

The K selector therefore failed badly and cannot support Voynich hand-count inference.

## Training diagnostics

- final 40-step HOG-MAE mean loss: `0.0488631284`, after a strong initial decline;
- final 30-step metric-learning mean loss: `1.1387253602`;
- the metric loss was effectively flat across the recorded intervals, so the smoke does not demonstrate effective NetRVLAD fine-tuning. Much of the retrieval signal may come from the HOG-MAE representation and aggregation rather than the metric stage.

## Post-run implementation audit

The smoke met its infrastructure purpose and corrected the v1.3 matrix-reuse defect. However, it was not sufficiently faithful to the frozen v1.4 full P1 specification to authorize a formal launch:

1. Morphological erosion/dilation augmentations declared in v1.4 were not applied by the smoke runner.
2. PCA was reduced to 64 dimensions because the miniature train panel could not support the frozen 512-dimensional setting; this smoke-only deviation was not explicitly recorded before execution.
3. Training used 160 MAE updates and 120 metric updates rather than a scaled epoch schedule corresponding to the public 200-epoch pretraining and 30-epoch fine-tuning configurations.
4. The metric stage used public-style SIFT-cluster pseudo-labels, but the short run and flat loss leave its effectiveness unresolved.
5. Perturbation retention and the full 199+ permutation programme were not part of this smoke.

## Decision

Do not open Voynich and do not treat v1.4 as a calibration pass. The result supports continuing with handwriting-specific retrieval and strongly disfavors returning DINOv3 to primary status, but a prospectively frozen v1.5 implementation is required before a larger external run.

v1.5 must:

- implement the declared morphology augmentations;
- specify a scaled but substantive pretraining and fine-tuning schedule before metrics;
- use 512-dimensional PCA in the full panel;
- retain writer-disjoint train/validation/test partitions and one-time terminal testing;
- retain exact matrix persistence and reuse;
- include acquisition, ink and combined nuisance comparisons;
- run at least 199 permutations and the complete K=2–10 calibration programme;
- reserve HisFrag20 as the independent terminal cross-corpus confirmation.
