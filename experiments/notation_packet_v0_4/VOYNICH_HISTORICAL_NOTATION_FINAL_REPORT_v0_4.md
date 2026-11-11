# Voynich historical-notation programme — corrected v0.4 result

**Date:** 2026-07-24  
**Status:** completed external calibration and sealed target adjudication  
**Repository branch:** `experiment/voynich-notation-historical-calibration-v0.4-20260724`  
**Formal verdict:** **ABSTAIN — Voynich lies outside the calibrated historical and control families**

## 1. External calibration

The corrected intake used exactly 2,400 canonical Ammerbach annotations: 1,200 neutral `bookA` and 1,200 `bookB`. Jupyter checkpoint duplicates and conflicting image-side metadata were excluded. The GABC corpus contributed Aquitanian and square-neume sources; unclassified GABC files were excluded from the preregistered tasks rather than treated as negative controls.

The externally selected representation was `flattened`.

| Measure | Corrected result |
|---|---:|
| Ensemble historical-notation ROC AUC | 0.9997 |
| Logistic ROC AUC | 0.9998 |
| Random-forest ROC AUC | 0.9996 |
| Calibrated balanced accuracy | 0.9579 |
| Organ-tablature recall | 1.0000 |
| Aquitanian-neume recall | 0.9200 |
| Square-neume recall | 0.9474 |
| Six-family macro-F1 | 0.8728 |
| Balanced-sensitivity AUC | 0.9959 |
| Balanced-sensitivity macro-F1 | 0.8769 |

The external gate and historical-family gate both passed. The pipeline can therefore recognise these known notation systems under manuscript-grouped holdout.

## 2. Sealed Voynich result

The sealed target comprised 346 fixed 48-token windows from 226 folios. The target was opened only after the corrected external gate passed.

| Formal `flattened` output | Result |
|---|---:|
| Mean historical-notation probability | 0.2454 |
| Median historical-notation probability | 0.1958 |
| Fraction above external threshold | 0.7861 |
| Mean organ-tablature probability | 0.0157 |
| Mean Aquitanian-neume probability | 0.0018 |
| Mean square-neume probability | 0.0398 |
| Mean procedural-synthetic probability | 0.8331 |
| Windows labelled procedural-synthetic | 336/346 |
| Cross-representation family agreement | 0.9538 |
| Windows beyond nearest-family 95th-percentile distance | 0.9971 |
| Median nearest-family distance percentile | 1.0000 |

The discriminative classifier consistently points toward the synthetic procedural family, not organ tablature or neumes. That tendency survives removing all explicit event-length features:

| Post-hoc feature set | External AUC | Family macro-F1 | Procedural windows | Mean historical probability |
|---|---:|---:|---:|---:|
| No length features | 0.9997 | 0.8488 | 253/346 | 0.1609 |
| No length or character-class features | 0.9997 | 0.8395 | 269/346 | 0.1905 |
| Sequence/entropy features only | 0.9997 | 0.8335 | 271/346 | 0.1550 |

However, **99.7% of Voynich windows lie beyond the nearest calibrated family's group-held-out 95th-percentile covariance distance**. The median distance percentile is 1.000. The target is therefore out of distribution. The classifier's high procedural probability is a directional resemblance, not a calibrated family assignment.

The apparent 78.6% threshold exceedance does not override the OOD failure. The broad threshold was calibrated among known external families; applying it far outside their support is not valid confirmatory evidence.

## 3. Scientific conclusion

The external experiment establishes that the feature pipeline can recognise real alphabetic organ tablature and neumatic notation. It does **not** place Voynichese inside either family.

The defensible conclusion is:

> Voynichese retains weakly stateful, learnable packet-like organisation, but it is not recognisably distributed like the calibrated historical musical notations. Relative to the available alternatives it points toward procedural synthetic structure, while remaining too far outside every calibrated family for a positive classification.

Accordingly:

- **literal musical notation:** not supported;
- **known neumatic or organ-tablature family:** rejected by distributional transfer at this resolution;
- **broad procedural or operational notation:** still viable but not identified;
- **specific synthetic-procedural classification:** abstain because of OOD;
- **next load-bearing evidence:** fifteenth-century German organ/lute surface transcriptions and more realistic non-musical procedural controls.

## 4. Corrections and audit trail

The first v0.4 external artifact is superseded. It accidentally included four Jupyter checkpoint duplicate annotations, misjoined CSV metadata across split-local numeric filenames, and treated unclassified GABC material as a negative class. The corrected artifact uses canonical CSV rows, exact 1,200/1,200 book balance, and preregistered class exclusions.

The earlier v0.2 absolute codelength correction remains in force: its `m_core` representation was lossy, although the matched HMM-minus-IID sequence contrast survived.
