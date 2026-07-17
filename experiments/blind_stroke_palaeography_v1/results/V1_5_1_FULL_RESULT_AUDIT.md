# SAGHOG v1.5.1 full-run result audit

**Audit date:** 2026-07-17  
**Status:** completed external-result audit; no Voynich data opened  
**Source job:** `6a5a1540d216bd6f3a1fb177`  
**Hardware:** A100-SXM4-80GB  
**Launcher:** `experiments/blind_stroke_palaeography_v1/code/run_saghog_v1_5_1.py`  
**Launcher commit:** `f376ee2a560dbbd1a0d2a3f06402cc70ec48b556`  
**Assembled v1.5.1 source SHA-256:** `fd8f93893a488b59d41eba4395de82e5690ebb491bc8bbe6c1de581a2884cdd8`  
**Upstream SAGHOG code commit:** `123cf0f306f105a46edbe8def06f49b54e64832e`  
**External archive MD5:** `e5ba2c7049bfb1453946233f681e4d53`

## Recovery and integrity

The complete job log was fetched read-only. It contained one `SAGHOG_V15_RESULT` record and 873 contiguous `V15_BUNDLE_CHUNK` records numbered 0–872.

- Bundle type: gzip-compressed TAR
- Bundle bytes: 2,617,073
- Bundle SHA-256: `7cdeb84c9b533e1d14f89a5102f4c8f050bbb9fd260fce06a5f5be27a1e339fa`
- `V15_BUNDLE_BEGIN` hash/size/chunk count: verified
- `V15_BUNDLE_END` hash: verified
- TAR members: `result.json`, `writer_split.json`, `exact_features.npz`

The bundle was reconstructed from the completed log, decoded with strict base64 validation, hash-verified, opened, and read end to end.

## Dataset and split audit

- Writers: 394
- Physical pages: 1,182
- Patches: 151,296
- Train writers/pages: 275 / 825
- Validation writers/pages: 59 / 177
- Terminal-test writers/pages: 60 / 180
- Writer duplicates within split: none
- Writer overlap between train, validation and test: none
- Every feature array is finite.
- Feature dimension: 512

The seal flags in the result are all false: `davis_labels_loaded=false`, `voynich_opened=false`, `f115r_loaded=false`.

## Selected model and representation

- Selected checkpoint: metric-learning step 500
- Checkpoint validation raw mAP: 0.3520080111116845
- Selected representation: `resid_combined`
- Selected-representation validation mAP: 0.38410194015646143
- Selected-representation terminal-test mAP: 0.4107341042252194
- Terminal-test top-1: 0.39444444444444443
- Terminal-test top-5: 0.6333333333333333
- Eligible terminal-test queries: 180

The step-500 checkpoint was preserved rather than overwritten by later validation deterioration.

## Validation retrieval

| Representation | mAP | top-1 | top-5 |
|---|---:|---:|---:|
| raw | 0.29751631991776206 | 0.2542372881355932 | 0.5423728813559322 |
| residualized acquisition | 0.3754008770885403 | 0.3728813559322034 | 0.6045197740112994 |
| residualized ink | 0.3192494637580343 | 0.3050847457627119 | 0.559322033898305 |
| residualized combined | 0.38410194015646143 | 0.3785310734463277 | 0.6497175141242938 |

## Terminal-test retrieval

| Representation | mAP | top-1 | top-5 |
|---|---:|---:|---:|
| raw | 0.32235582526308526 | 0.3 | 0.5111111111111111 |
| residualized acquisition | 0.3973870413929802 | 0.4 | 0.6111111111111112 |
| residualized ink | 0.32846542393624034 | 0.3055555555555556 | 0.5277777777777778 |
| residualized combined | 0.4107341042252194 | 0.39444444444444443 | 0.6333333333333333 |

## Nuisance baselines

| Nuisance representation | mAP | top-1 | top-5 |
|---|---:|---:|---:|
| acquisition | 0.3053187055502651 | 0.3111111111111111 | 0.5222222222222223 |
| ink | 0.21899506475652689 | 0.19444444444444445 | 0.4166666666666667 |
| combined acquisition + ink | 0.2954052001550606 | 0.3333333333333333 | 0.5277777777777778 |

Selected retrieval exceeds acquisition nuisance by 0.10541539867495431, but the ratio is only 1.3452634796317764, below the frozen 1.5× threshold.

## Permutation test

- Permutations: 199
- p-value: 0.005
- Null mean mAP: 0.037652080011720775
- Null SD: 0.006856527902881665
- Null permutations equalling or exceeding the observed mAP: 0 of 199

## Perturbation retention

| Perturbation | mAP | retention |
|---|---:|---:|
| contrast | 0.4557690626956966 | 1.1096450428810338 |
| scale | 0.4462809854506031 | 1.0865447520907399 |
| erosion | 0.35914970007540414 | 0.8744092501227272 |
| dilation | 0.35780823871133455 | 0.8711432409204963 |
| translation | 0.444982183198795 | 1.0833826035414782 |

All five implemented perturbations exceed the frozen 0.80 retention threshold.

## Synthetic K calibration

- Panels: 45 total; five repetitions for each true K from 2 through 10
- Exact-K recovery: 0.13333333333333333
- Within ±1 recovery: 0.2222222222222222
- Selected K=2 in 43 of 45 panels
- Selected K=3 in 2 of 45 panels
- No larger K was selected

This is a severe collapse toward K=2 and is not usable for hand enumeration.

## Frozen gate decisions

| Gate | Decision |
|---|---|
| mAP exceeds acquisition nuisance by at least 0.05 | PASS |
| mAP is at least 1.5× acquisition nuisance | FAIL |
| permutation p ≤ 0.01 with at least 199 permutations | PASS |
| all implemented perturbation retentions ≥ 0.80 | PASS |
| exact K ≥ 0.70 | FAIL |
| K within ±1 ≥ 0.90 | FAIL |
| all implemented gates | FAIL |
| held-out-family performance ≥ 0.70 where applicable | NOT EVALUATED |
| false-discrete rate ≤ 0.05 | NOT EVALUATED |
| null abstention ≥ 0.95 | NOT EVALUATED |
| at least 40 panels per mechanism | NOT ESTABLISHED; only five repetitions per true K were run |

No retrospective gate weakening is permitted.

## Output hashes

### Reconstructed bundle members

- `result.json`: 9,536 bytes; SHA-256 `38572ebf1997dd7b3b859cafdd782612c2bc3fd09a253d17930870fa8d758a3a`
- `writer_split.json`: 5,888 bytes; SHA-256 `aa111caf6db8f1c3738ccbbc8c20b518c671e70520822379d75f41be4180d296`
- `exact_features.npz`: 2,612,047 bytes; SHA-256 `dd47d63b635ea2f4722920221b146491b59f82b63f5b2fbffe6c12c6c06f9a52`

### Checkpoint reported by the emitted result

- `saghog_v15_best.pt`: 115,400,460 bytes; SHA-256 `9b543dbc13600a8a32ca04e7794541d8e184c0cbcd793e1b9492d37e03445d09`

The checkpoint was not included in the emitted TAR bundle, so its bytes were not independently re-hashed during this audit.

## Independent recomputation

Writer retrieval from `exact_features.npz`, the 199-permutation test and all 45 K-calibration panels were independently recomputed. Selected/raw retrieval, permutation and K outputs reproduce exactly. Acquisition-only and ink-only mAP differ from the stored values by 4.63e-5 and 4.17e-6 respectively under float64 recomputation; top-1/top-5 and the combined nuisance result are identical. This is consistent with tie-sensitive ordering under different floating-point precision and does not affect any decision.

## Implementation anomalies

1. `result.json` has a self-hash ordering defect. The archived file internally declares an earlier 8,978-byte version with SHA-256 `e58955ce4800efcaca49b9c692eb7f0f9f65ad1ebc84d0290e052547e60c9f47`. The emitted `SAGHOG_V15_RESULT` correctly declares the final archived 9,536-byte file and SHA-256 `38572ebf...`. All scientific fields are identical; only the nested self-record differs.
2. The selected checkpoint is named and hashed in the result but is absent from the recovery bundle. It must be persisted separately before any reuse or independent model-level audit.
3. Several frozen confirmatory gates were not implemented in this run: held-out-family transfer, false-discrete rate, null abstention, and the required panel count per mechanism. They cannot be treated as passes.

## Scientific decision

**Category B: writer retrieval is significant and perturbation-robust, but nuisance-ratio and K-recovery gates fail.**

This is not a validated writer-identification model and cannot support independent hand recovery or writer-count claims. The result is sufficiently positive to justify a prospectively frozen v1.6 test of continuous, nuisance-resistant page/fragment style similarity. Voynich remains sealed until that validation is completed and frozen.