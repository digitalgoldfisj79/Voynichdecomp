# P1 SAGHOG v1.5.1 — full external Historical-WI result

**Date:** 2026-07-17  
**HF job:** `6a5a1540d216bd6f3a1fb177`  
**Status:** completed; P1 closed as a failed formal calibration branch  
**Schema:** `blind-pal-saghog-v1.5-full`  
**Voynich opened:** no  
**Davis labels loaded:** no  
**f115r boundary loaded:** no

## Provenance and execution

- Historical-WI writers: 394.
- Writer-disjoint split: 275 training, 59 validation, 60 terminal test writers.
- Physical pages: 1,182.
- Extracted handwriting patches: 151,296.
- HOG-MAE pretraining: 5,000 steps.
- Validation-selected metric checkpoint: step 500.
- Validation raw mAP at selected checkpoint: 0.3520080111.
- Selected representation: `resid_combined`.
- Formal permutation count: 199.
- Synthetic-K panels: 45, covering K=2–10 with five replicates per K.

## Terminal test retrieval

| Representation | mAP | Top-1 | Top-5 |
|---|---:|---:|---:|
| raw | 0.322356 | 0.300000 | 0.511111 |
| residualized against acquisition | 0.397387 | 0.400000 | 0.611111 |
| residualized against ink | 0.328465 | 0.305556 | 0.527778 |
| residualized against acquisition + ink (selected) | **0.410734** | **0.394444** | **0.633333** |

## Nuisance baselines

| Baseline | mAP | Top-1 | Top-5 |
|---|---:|---:|---:|
| acquisition | **0.305319** | 0.311111 | 0.522222 |
| ink | 0.218995 | 0.194444 | 0.416667 |
| combined | 0.295405 | 0.333333 | 0.527778 |

The selected handwriting representation exceeds the acquisition baseline by 0.105415 mAP, satisfying the frozen absolute-margin requirement of +0.05. Its ratio to acquisition is 1.345, below the frozen 1.5× requirement.

## Significance and robustness

- Permutation p-value: **0.005** with 199 permutations.
- Permutation-null mAP: mean 0.037652, SD 0.006857.
- Contrast retention: 1.109645.
- Dilation retention: 0.871143.
- Erosion retention: 0.874409.
- Scale retention: 1.086545.
- Translation retention: 1.083383.

All reported perturbation retentions exceed the frozen 0.80 threshold.

## Synthetic-K calibration

- Panels: 45.
- Exact-K recovery: **0.133333**; frozen requirement 0.70.
- Within-one recovery: **0.222222**; frozen requirement 0.90.
- The selector overwhelmingly collapsed to K=2 for true K above two.

## Frozen gate decision

| Gate | Result |
|---|---|
| mAP exceeds acquisition by at least 0.05 | PASS |
| mAP at least 1.5× acquisition | FAIL |
| permutation p ≤ 0.01 with at least 199 permutations | PASS |
| perturbation retention ≥ 0.80 | PASS |
| exact-K ≥ 0.70 | FAIL |
| within-one-K ≥ 0.90 | FAIL |
| all gates | **FAIL** |

## Interpretation and terminal decision

The public-code SAGHOG reproduction learned a statistically significant and perturbation-robust writer signal. Residualization materially improved writer retrieval, and the terminal mAP is substantially better than the earlier frozen DINOv3 branch. However, it did not separate writer identity from acquisition nuisance strongly enough under the preregistered ratio gate, and its representation did not support reliable recovery of the number of writers.

Under the prospective v1.5 decision rule, this exact P1 configuration is closed. It will not be tuned against the terminal Historical-WI test set and does not unlock any Voynich analysis. The next prospectively distinct branch is P2: foreground-token self-supervised ViT features with VLAD aggregation, whose checkpoint, source digest, token layer, foreground rule, aggregation parameters and evaluation runner must be frozen before any P2 metric.

## Result artifacts

- Full log bundle SHA-256: `7cdeb84c9b533e1d14f89a5102f4c8f050bbb9fd260fce06a5f5be27a1e339fa`.
- `exact_features.npz`: SHA-256 `dd47d63b635ea2f4722920221b146491b59f82b63f5b2fbffe6c12c6c06f9a52`.
- `result.json`: SHA-256 `38572ebf1997dd7b3b859cafdd782612c2bc3fd09a253d17930870fa8d758a3a`.
- `saghog_v15_best.pt`: SHA-256 `9b543dbc13600a8a32ca04e7794541d8e184c0cbcd793e1b9492d37e03445d09`.
- `writer_split.json`: SHA-256 `aa111caf6db8f1c3738ccbbc8c20b518c671e70520822379d75f41be4180d296`.
