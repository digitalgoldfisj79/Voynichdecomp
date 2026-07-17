# v1.6 stage-0 characterization of existing terminal features

Date: 2026-07-17

## Scope

This is a secondary characterization of the already observed v1.5.1 terminal-test matrix. It is not an independent external validation and cannot authorize opening Voynich.

- Source job: `6a5a1540d216bd6f3a1fb177`
- Verified bundle SHA-256: `7cdeb84c9b533e1d14f89a5102f4c8f050bbb9fd260fce06a5f5be27a1e339fa`
- Stage-0 code commit: `bcb05a4a0170ef5637f7f6877ccf0dab8321059d`
- Stage-0 job: `6a5a4d8ed216bd6f3a1fb96c`
- Writer-disjoint nuisance-audit code commit: `0ccb147ae7b2d692440f3c99af4afc8eb4a77338`
- Writer-disjoint nuisance-audit job: `6a5a4ebfd216bd6f3a1fb985`
- Bootstrap replicates: 2,000
- Label permutations: 999
- Voynich opened: false
- Davis labels loaded: false
- f115r loaded: false

## Full terminal matrix

The matrix contains 180 physical-page rows from 60 writers, with three pages per writer.

- positive different-row same-writer pairs: 180
- different-writer pairs: 15,930
- positive-pair prevalence: 0.01117

### Pair-discrimination metrics

| Representation | ROC-AUC | Average precision | AP lift | EER |
|---|---:|---:|---:|---:|
| selected `resid_combined` | 0.8810 | 0.3044 | 0.2932 | 0.1948 |
| raw SAGHOG | 0.8513 | 0.1894 | 0.1782 | 0.2378 |
| acquisition nuisance | 0.7534 | 0.1546 | 0.1434 | 0.3028 |
| ink nuisance | 0.7621 | 0.0650 | 0.0538 | 0.3110 |
| combined nuisance, standardized on all terminal pages | 0.9002 | 0.3169 | 0.3057 | 0.1888 |

Selected representation writer-cluster 95% intervals:

- ROC-AUC: 0.8479–0.9114
- average precision: 0.2249–0.4043
- AP lift: 0.2136–0.3930
- EER: 0.1611–0.2333

The selected representation is strongly non-random under 999 label permutations:

- observed AUC: 0.8810
- null mean: 0.4994
- null SD: 0.0206
- p-value: 0.001

### Descriptive calibration

A logistic calibrator was fitted on 30 writers and evaluated on a disjoint 30-writer subset.

- Brier score: 0.02005
- prevalence-only Brier score: 0.02197
- 10-bin ECE: 0.00467

This shows that the selected score contains calibratable writer information. It does not show that the information is handwriting-specific.

## Writer-disjoint nuisance audit

The initial standardized combined-nuisance result fitted its scaler on all terminal pages. That was deliberately treated as a stress test, not as an admissible independent comparison. A second audit prospectively froze the original 30/30 writer split, fitted every nuisance scaler on the calibration writers only, and evaluated on the untouched 30 writers.

Evaluation subset:

- writers: 30
- pages: 90
- positive pairs: 90
- pair prevalence: 0.02247

### Writer-disjoint pair metrics

| Representation | ROC-AUC | Average precision | AP lift | EER |
|---|---:|---:|---:|---:|
| selected `resid_combined` | 0.8595 | 0.3414 | 0.3189 | 0.2163 |
| raw SAGHOG | 0.8287 | 0.2274 | 0.2050 | 0.2579 |
| acquisition, calibration-scaled | 0.8557 | 0.2726 | 0.2501 | 0.2328 |
| ink, calibration-scaled | 0.8345 | 0.1701 | 0.1477 | 0.2234 |
| combined nuisance, unscaled | 0.7932 | 0.2147 | 0.1923 | 0.2650 |
| combined nuisance, calibration-scaled | **0.8918** | **0.3422** | **0.3197** | **0.1969** |

### Writer-disjoint retrieval

| Representation | mAP | Top-1 | Top-5 |
|---|---:|---:|---:|
| selected `resid_combined` | 0.4517 | 0.4333 | 0.7667 |
| raw SAGHOG | 0.3764 | 0.3556 | 0.6556 |
| acquisition, calibration-scaled | 0.4114 | 0.3778 | 0.6667 |
| ink, calibration-scaled | 0.3427 | 0.3444 | 0.6556 |
| combined nuisance, unscaled | 0.3500 | 0.3556 | 0.6222 |
| combined nuisance, calibration-scaled | **0.5074** | **0.4889** | **0.7778** |

### Bootstrap comparison

Selected AUC minus calibration-scaled combined-nuisance AUC:

- point difference: −0.0323
- writer-cluster 95% interval: **−0.0865 to +0.0227**

Selected AUC minus calibration-scaled acquisition AUC:

- point difference: +0.0037
- writer-cluster 95% interval: −0.0502 to +0.0554

Selected AUC minus unscaled combined-nuisance AUC:

- point difference: +0.0663
- writer-cluster 95% interval: +0.0029 to +0.1299

The scientific conclusion depends on treating nuisance coordinates comparably. Once nuisance variables are standardized using disjoint calibration writers, the selected representation does not beat the combined nuisance vector.

## Interpretation

The representation clearly encodes a signal associated with writer labels:

- pair AUC is high;
- AP is far above prevalence;
- retrieval is significant;
- calibration is better than a prevalence-only predictor;
- the signal survives writer-cluster uncertainty.

However, the existing corpus does **not** establish that the signal is handwriting-specific. A simple combined acquisition-and-ink descriptor can match or exceed both pair discrimination and retrieval once its coordinates are scaled using disjoint calibration writers.

Therefore:

1. **Writer identification remains failed under the original frozen gates.**
2. **Discrete hand enumeration remains decisively failed.**
3. **Continuous writing-style similarity remains only a candidate, not a validated instrument.**
4. The candidate must beat calibration-fitted nuisance baselines on HisFrag20 and a second corpus with richer acquisition/letter/date metadata.
5. If the external primary representation again fails to beat combined nuisance by the frozen margin, direct Voynich application is prohibited and SAGHOG P1 should be closed or restricted to an auxiliary feature.

The appropriate current classification remains Category B for the completed v1.5.1 result, but the stage-0 evidence substantially increases the probability that the programme will move to Category C after external validation.
