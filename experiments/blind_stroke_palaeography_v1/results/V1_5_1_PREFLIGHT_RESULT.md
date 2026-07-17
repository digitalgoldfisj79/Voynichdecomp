# v1.5.1 SAGHOG preflight result

**Date:** 2026-07-17  
**HF job:** `6a5a1378bee6ee1cf4ecd3e3`  
**Status:** completed  
**Purpose:** implementation and end-to-end preflight; not confirmatory external calibration  
**Voynich opened:** no  
**Davis labels loaded:** no  
**f115r boundary loaded:** no

## Execution audit

- Hash-verified v1.5.1 source assembled successfully.
- Historical-WI colour archive downloaded and verified.
- Writer-disjoint split: 56 train, 12 validation, 12 terminal test writers.
- 240 physical pages and 15,360 handwriting patches processed.
- HOG-MAE pretraining, NetRVLAD metric learning, validation checkpoint selection, exact feature persistence, nuisance residualization, permutation test, perturbation tests, and K=2–10 calibration completed.
- Result bundle SHA-256: `3a102427aff8c2312ce0e4c509aac231d2306b7a572fe7b3c57bf51fc563903a`.

## Miniature preflight metrics

Selected representation: `resid_combined`.

- Validation mAP: 0.459817.
- Terminal test mAP: 0.299835.
- Raw terminal test mAP: 0.279277.
- Acquisition nuisance mAP: 0.572803.
- Permutation: p=0.05 with 19 permutations.
- Perturbation retention: contrast 0.578; dilation 1.166; erosion 1.557; scale 0.993; translation 0.926.
- Synthetic K: exact 0.222; within-one 0.333 across nine single-replicate panels.

## Interpretation

The preflight validates the corrected implementation route. It does not pass the scientific gates, but it used only 40 HOG-MAE steps, 40 metric-learning steps, 19 permutations, and one panel per K. These settings were intentionally miniature and are inadequate for a scientific verdict.

The high acquisition nuisance score remains a material risk. The formal run must determine whether substantive training, complete permutation testing, repeated panels, and independent HisFrag20 transfer can produce writer signal that exceeds this nuisance baseline. No Voynich phase is unlocked by this preflight.

## Decision

Proceed to the preregistered full external Historical-WI v1.5.1 run. Treat the full run as terminal for this exact P1 configuration: if it fails the frozen gates, close P1 rather than tuning against the terminal test set.
