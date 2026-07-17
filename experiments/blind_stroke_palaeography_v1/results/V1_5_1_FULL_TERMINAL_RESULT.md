# SAGHOG v1.5.1 full Historical-WI terminal result

**Date:** 2026-07-17  
**HF job:** `6a5a1540d216bd6f3a1fb177`  
**Status:** completed; P1 closed for this exact configuration  
**Voynich opened:** no  
**Davis labels loaded:** no  
**f115r boundary loaded:** no

## Execution

- 394 writers: 275 train, 59 validation, 60 terminal test.
- 1,182 physical pages.
- 151,296 handwriting patches.
- HOG-MAE: 5,000 steps.
- Best metric-learning checkpoint selected prospectively on validation at step 500; validation raw mAP 0.352008.
- Selected representation: `resid_combined`.
- Terminal result bundle SHA-256: `ed9b479b563d4ff748a889381ec305dd7afc0965ec4bbc4b7b7c9ccd67e29a4a`.

## Terminal test results

- Selected writer representation: mAP 0.410734; top-1 0.394444; top-5 0.633333; 180 eligible page queries.
- Raw writer representation: mAP 0.322356.
- Acquisition nuisance baseline: mAP 0.305319.
- Combined nuisance baseline: mAP 0.295405.
- Absolute selected-over-acquisition margin: +0.105415 — pass against frozen +0.05 gate.
- Selected/acquisition ratio: 1.3453 — fail against frozen 1.5× gate.

## Statistical and robustness gates

- Permutation: p=0.005, 199 permutations; null mean 0.037652, null SD 0.006857 — pass.
- Perturbation retention: contrast 1.109645; dilation 0.871143; erosion 0.874409; scale 1.086545; translation 1.083383 — all pass frozen >=0.80 gate.
- Synthetic K calibration: 45 panels; exact-K 0.133333; within-one 0.222222 — fail frozen exact >=0.70 and within-one >=0.90 gates.

## Frozen criteria

- absolute_over_acquisition: pass
- ratio_over_acquisition: fail
- permutation: pass
- perturbation: pass
- exact_k: fail
- within_one_k: fail
- all_pass: false

## Interpretation and decision

The handwriting-specific representation contains a real and robust writer-identification signal. It materially outperforms the acquisition nuisance baseline in absolute terms and is highly significant under permutation. However, it does not clear the preregistered relative-nuisance criterion and, critically, it cannot recover the known number of writers in synthetic K=2–10 panels. The representation is therefore useful for pairwise similarity/retrieval but is not validated as a discrete hand-counting instrument.

Per the prospectively frozen rule, close P1 for this exact SAGHOG configuration rather than tuning against the terminal test set. Do not unlock Voynich Phase I. P2 foreground-token ViT + VLAD and the independent classical/HTR branches remain eligible for external calibration under separately frozen specifications.