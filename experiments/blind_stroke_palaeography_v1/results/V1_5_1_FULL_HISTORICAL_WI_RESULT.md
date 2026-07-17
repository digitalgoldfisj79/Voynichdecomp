# v1.5.1 full Historical-WI result — terminal SAGHOG P1

**Date:** 2026-07-17  
**HF job:** `6a5a1540d216bd6f3a1fb177`  
**Status:** completed  
**Schema:** `blind-pal-saghog-v1.5-full`  
**Voynich opened:** no  
**Davis labels loaded:** no  
**f115r boundary loaded:** no

## Execution

- 394 writers
- 1,182 physical pages
- 151,296 patches
- selected checkpoint: metric-learning step 500
- selected checkpoint validation raw mAP: 0.352008
- selected representation: `resid_combined`
- exact persisted feature matrix SHA-256: `dd47d63b635ea2f4722920221b146491b59f82b63f5b2fbffe6c12c6c06f9a52`
- result JSON SHA-256: `38572ebf1997dd7b3b859cafdd782612c2bc3fd09a253d17930870fa8d758a3b`
- checkpoint SHA-256: `9b543dbc13600a8a32ca04e7794541d8e184c0cbcd793e1b9492d37e03445d09`
- bundle SHA-256: `7cdeb84c9b533e1d14f89a5102f4c8f050bbb9fd260fce06a5f5be27a1e339fa`

## Terminal retrieval

| Representation | mAP | Top-1 | Top-5 |
|---|---:|---:|---:|
| raw | 0.322356 | 0.300000 | 0.511111 |
| residualized against acquisition | 0.397387 | 0.400000 | 0.611111 |
| residualized against ink | 0.328465 | 0.305556 | 0.527778 |
| residualized against combined nuisance — selected | **0.410734** | **0.394444** | **0.633333** |

## Nuisance controls

| Baseline | mAP | Top-1 | Top-5 |
|---|---:|---:|---:|
| acquisition | 0.305319 | 0.311111 | 0.522222 |
| ink | 0.218995 | 0.194444 | 0.416667 |
| combined | 0.295405 | 0.333333 | 0.527778 |

The selected writer representation exceeds acquisition nuisance by 0.105415 absolute, but only by 1.3453×. It therefore passes the frozen absolute-margin gate and fails the frozen 1.5× ratio gate.

## Permutation

- 199 permutations
- observed mAP: 0.410734
- null mean: 0.037652
- null SD: 0.006857
- p = 0.005
- gate p ≤ 0.01: **pass**

## Perturbation retention

| Perturbation | mAP | Retention |
|---|---:|---:|
| contrast | 0.455769 | 1.109645 |
| dilation | 0.357808 | 0.871143 |
| erosion | 0.359150 | 0.874409 |
| scale | 0.446281 | 1.086545 |
| translation | 0.444982 | 1.083383 |

All frozen retention thresholds ≥0.80 pass.

## Synthetic K=2–10 calibration

- 45 panels, five replicates per true K
- exact K recovery: 0.133333
- within-one recovery: 0.222222
- frozen exact gate ≥0.70: **fail**
- frozen within-one gate ≥0.90: **fail**

The selector collapsed overwhelmingly to K=2, including for most panels generated with larger true K.

## Frozen gate decision

- absolute margin over acquisition ≥0.05: **pass**
- ratio over acquisition ≥1.5×: **fail**
- permutation p≤0.01 with 199 permutations: **pass**
- perturbation retention ≥0.80: **pass**
- exact K recovery ≥0.70: **fail**
- within-one K recovery ≥0.90: **fail**
- all gates: **fail**

## Interpretation and programme decision

SAGHOG P1 learned statistically significant and perturbation-stable writer information. It materially outperformed the acquisition nuisance baseline in absolute terms. It did not establish the required nuisance separation ratio and, more decisively, failed the discrete-hand-number calibration badly. The representation is useful evidence that writer signal is recoverable, but it is not a validated arbiter of K and cannot unlock Voynich analysis.

Under the preregistered terminal-test rule, this exact P1 configuration is closed. It will not be tuned against this terminal test set. The next prospective branch is P2: foreground-token self-supervised ViT features with VLAD aggregation, frozen before any metric. Voynich images, Davis labels and the f115r boundary remain sealed.