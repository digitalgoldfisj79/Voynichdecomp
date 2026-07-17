# v1.3 end-to-end smoke result and stop condition

**Date:** 2026-07-17  
**Job:** `6a5a06f3bee6ee1cf4ecd256`  
**Status:** execution completed; scientific smoke failed; v1.3 closed before full calibration  
**Voynich Phase I opened:** no  
**Davis labels loaded:** no  
**f115r boundary loaded:** no

## Execution result

The complete Historical-WI smoke ran end to end on an A100 using the preregistered v1.3 DINOv3 ViT-7B/16 bucket substitution. The bucket was copied to local NVMe, all six weight shards were hashed, the model loaded locally, and both DINOv3 and historical-TrOCR inference completed. Fold-local evaluation, retrieval, provisional permutations, synthetic-K calibration and result serialization all executed.

Panel: 20 writers × 3 physical pages = 60 retrieval items; one colour derivative per physical page; maximum two tiles per item; three smoke permutations.

## Provisional smoke numbers

- Best non-nuisance representation: `dino`
- DINO mAP: `0.0696202099`
- DINO top-1: `0.0166666667`
- DINO top-5: `0.1166666667`
- Nuisance mAP: `0.5424913498`
- Nuisance top-1: `0.5833333333`
- Nuisance top-5: `0.8`
- Absolute mAP over nuisance: `-0.4728711399`
- mAP ratio over nuisance: `0.1283342304`
- Synthetic-K exact rate: `0.25`
- Synthetic-K within-one rate: `0.375`
- Smoke permutation p: `1.0` with only three permutations
- Partial gate pass: `false`; every provisional gate false

These numbers are non-confirmatory because this was a deliberately small smoke panel, but they are sufficient to prohibit an immediate full calibration launch.

## Evaluation defect discovered after the smoke

The smoke also exposed a concrete implementation inconsistency in the frozen evaluator:

1. Retrieval mAP for each representation is computed on fold-local out-of-fold nuisance-residualized features inside `evaluate_foldlocal`.
2. The selected representation name is then mapped back to `base[proxy_name]`.
3. `permutation_p` and `synthetic_k_calibration` are run on that raw base representation rather than the exact fold-local out-of-fold representation whose mAP selected the model.
4. Consequently the permutation observed statistic and null features are not the same representation, and the K-calibration is not evaluating the selected fold-local representation.

The provisional permutation and K results are therefore invalid as calibration evidence. The retrieval and nuisance mAP values remain valid outputs of the smoke implementation, but their interpretation is limited by the very small panel and the exceptionally strong nuisance baseline.

## Stop decision

No full Historical-WI or HisFrag20 calibration will be launched under v1.3. Version 1.3 is closed. A new preregistered version must:

- persist and return the exact fold-local out-of-fold feature matrices for every candidate representation;
- run permutation and synthetic-K calibration on the exact selected out-of-fold representation;
- retain the same-page exclusion and writer-balanced physical-page folds;
- explicitly diagnose why whole-item nuisance features identify Historical-WI writers so strongly, including feature ablations and acquisition/source leakage checks;
- determine prospectively whether the nuisance gate is to remain unchanged, be applied after a narrower nuisance specification, or cause Historical-WI to be rejected as a primary calibration corpus;
- remain completely external to Voynich, Davis labels and f115r during this diagnostic and preregistration stage.

No scientific inference about Voynich hands is permitted from this smoke.