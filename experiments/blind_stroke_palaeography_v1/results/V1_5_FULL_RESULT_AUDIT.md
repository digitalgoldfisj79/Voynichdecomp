# SAGHOG v1.5 full-run audit

Status: **frozen result record**  
Date: 2026-07-17  
Job: `6a5a1540d216bd6f3a1fb177`  
Launcher: `experiments/blind_stroke_palaeography_v1/code/run_saghog_v1_5_1.py`  
Launcher commit: `f376ee2a560dbbd1a0d2a3f06402cc70ec48b556`  
Hardware: A100-SXM4-80GB (`a100-large`)  
Runtime: 3,805 seconds  
Schema: `blind-pal-saghog-v1.5-full`  
Seed: `20260718`

## Terminal decision

**Category B: writer retrieval is statistically significant and perturbation-robust, but the representation fails the nuisance-ratio and discrete-K gates.**

Consequences:

- This is **not** a validated writer-identification model under the frozen v1.5 gates.
- It does **not** independently recover scribal hands or support a claim about the number of hands.
- It is a plausible continuous handwriting-style similarity representation, but that use requires prospective v1.6 validation before any Voynich material is opened.
- Davis labels, Davis hand assignments, f115r boundary data, Voynich section labels and Currier labels remain sealed.

## Data and split

- Writers: 394
- Physical pages: 1,182
- Handwriting patches: 151,296
- Training writers: 275
- Validation writers: 59
- Terminal-test writers: 60
- Validation page vectors: 177
- Terminal-test page vectors: 180
- Representation dimension: 512

## Model selection

- Selected checkpoint step: **500**
- Raw validation mAP recorded at selected checkpoint: **0.3520080111**
- Selected representation: **`resid_combined`**
- Final validation mAP for selected representation: **0.3841019402**
- Terminal-test mAP for selected representation: **0.4107341042**

The step-500 checkpoint was correctly retained despite later deterioration in raw validation diagnostics.

## Validation results

| Representation | mAP | Top-1 | Top-5 |
|---|---:|---:|---:|
| raw | 0.2975163199 | 0.2542372881 | 0.5423728814 |
| resid_acquisition | 0.3754008771 | 0.3728813559 | 0.6045197740 |
| resid_ink | 0.3192494638 | 0.3050847458 | 0.5593220339 |
| resid_combined | **0.3841019402** | **0.3785310734** | **0.6497175141** |

## Terminal-test results

| Representation | mAP | Top-1 | Top-5 |
|---|---:|---:|---:|
| raw | 0.3223558253 | 0.3000000000 | 0.5111111111 |
| resid_acquisition | 0.3973870414 | 0.4000000000 | 0.6111111111 |
| resid_ink | 0.3284654239 | 0.3055555556 | 0.5277777778 |
| resid_combined | **0.4107341042** | **0.3944444444** | **0.6333333333** |

## Nuisance baselines

| Baseline | mAP | Top-1 | Top-5 |
|---|---:|---:|---:|
| acquisition | **0.3053187056** | 0.3111111111 | 0.5222222222 |
| ink | 0.2189950648 | 0.1944444444 | 0.4166666667 |
| combined | 0.2954052002 | 0.3333333333 | 0.5277777778 |

Comparison against acquisition nuisance:

- Absolute margin: `0.4107341042 - 0.3053187056 = 0.1054153987` — passes the required +0.05 gate.
- Ratio: `0.4107341042 / 0.3053187056 = 1.34554` — fails the required 1.5× gate.

## Permutation test

- Permutations: **199**
- Null mean mAP: **0.0376520800**
- Null standard deviation: **0.0068565279**
- p-value: **0.005**
- Frozen gate `p <= 0.01` with at least 199 permutations: **PASS**

## Perturbation retention

Retention is relative to the selected terminal-test representation.

| Perturbation | mAP | Retention | Gate ≥ 0.80 |
|---|---:|---:|---:|
| contrast | 0.4557690627 | 1.1096450429 | PASS |
| dilation | 0.3578082387 | 0.8711432409 | PASS |
| erosion | 0.3591497001 | 0.8744092501 | PASS |
| scale | 0.4462809855 | 1.0865447521 | PASS |
| translation | 0.4449821832 | 1.0833826035 | PASS |

All implemented perturbation gates pass.

## Synthetic K calibration

- Panels: **45**
- Exact-K recovery: **0.1333333333**
- K within ±1: **0.2222222222**
- Required exact-K ≥ 0.70: **FAIL**
- Required within ±1 ≥ 0.90: **FAIL**

The selector chose K=2 for 43 of 45 panels and K=3 for two panels. It is strongly biased toward low K and is not a valid hand-enumeration mechanism.

## Frozen gate decisions

| Gate | Decision |
|---|---|
| writer mAP exceeds acquisition nuisance by ≥0.05 | PASS |
| writer mAP ≥1.5× acquisition nuisance | **FAIL** |
| permutation p≤0.01 with ≥199 permutations | PASS |
| all perturbation retentions ≥0.80 | PASS |
| exact K≥0.70 | **FAIL** |
| K within ±1≥0.90 | **FAIL** |
| all frozen gates | **FAIL** |

Held-out-family, false-discrete and null-abstention gates were not established by this result and must not be treated as passed.

## Training diagnostics

- HOG-MAE final 100-step mean loss: `0.0294844926`
- Metric-learning final recorded check loss: `1.6219358804`

These are diagnostics only and do not alter the terminal decision.

## Bundle verification

The terminal logs contained 873 `V15_BUNDLE_CHUNK` records.

- Reconstructed bundle size: 2,617,073 bytes
- Declared SHA-256: `7cdeb84c9b533e1d14f89a5102f4c8f050bbb9fd260fce06a5f5be27a1e339fa`
- Recomputed SHA-256: `7cdeb84c9b533e1d14f89a5102f4c8f050bbb9fd260fce06a5f5be27a1e339fa`
- Verification: **PASS**

## Output hashes

Terminal `SAGHOG_V15_RESULT` record:

| File | Bytes | SHA-256 |
|---|---:|---|
| `exact_features.npz` | 2,612,047 | `dd47d63b635ea2f4722920221b146491b59f82b63f5b2fbffe6c12c6c06f9a52` |
| `result.json` | 9,536 | `38572ebf1997dd7b3b859cafdd782612c2bc3fd09a253d17930870fa8d758a3a` |
| `saghog_v15_best.pt` | 115,400,460 | `9b543dbc13600a8a32ca04e7794541d8e184c0cbcd793e1b9492d37e03445d09` |
| `writer_split.json` | 5,888 | `aa111caf6db8f1c3738ccbbc8c20b518c671e70520822379d75f41be4180d296` |

The extracted `exact_features.npz` contains:

- `train`: `(825, 512)`
- `val`: `(177, 512)`
- `test`: `(180, 512)`
- `test_selected`: `(180, 512)`
- writer arrays for all three splits
- acquisition nuisance features: `(180, 25)`
- ink nuisance features: `(180, 11)`

## Implementation anomaly

A non-scientific persistence anomaly is visible:

- The final emitted `SAGHOG_V15_RESULT` and the actual extracted file report `result.json` as 9,536 bytes with SHA-256 `38572e...`.
- Inside `result.json`, its self-referential `files.result.json` entry still records an earlier 8,978-byte version with SHA-256 `e58955...`.

This is consistent with computing the self-hash before the final JSON rewrite. It does not affect the metrics, feature matrices, split manifest, checkpoint selection or bundle verification, but future launchers should avoid self-referential file hashes or write them to a separate manifest.

No evidence of Voynich leakage is visible:

- `davis_labels_loaded=false`
- `f115r_loaded=false`
- `voynich_opened=false`

## Required next action

Freeze and execute v1.6 as a prospective validation of **continuous, nuisance-resistant page-style similarity**. Do not rerun v1.5, open Voynich, or claim writer identification or K recovery.