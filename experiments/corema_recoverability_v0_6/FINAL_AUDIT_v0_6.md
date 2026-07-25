# Final scientific audit — CoReMA procedural recoverability v0.6

**Date:** 2026-07-25  
**Branch:** `experiment/voynich-corema-recoverability-v0.6-20260725`  
**Formal verdict:** `CALIBRATION_FAILURE`  
**Target status:** Voynich transfer remained sealed.

## 1. Locked result

The corrected formal run used 29 manuscript groups, 4,636 procedural units and 370,980 labelled tokens. The lexical known-role gate failed at macro-F1 0.5721 against a frozen threshold of 0.60. The identity-neutral gate failed at structural-HMM macro-F1 0.3270 against a frozen threshold of 0.35, although its margin over the held-out majority baseline exceeded the required 0.10. The sequence-order gate passed decisively: the mean real-order advantage over within-recipe shuffling was 0.9268 bits/token and was positive in all five folds.

The gates were not revised after inspection. Downstream manuscript transfer was therefore inadmissible.

## 2. Failure classification

### 2.1 Ordinary sample-size insufficiency — not the primary explanation

The corpus is large at the token level and every primary role exceeded the frozen support rule. All 29 manuscripts entered grouped cross-validation. Broad recipe-type recovery was also strong when literal lexical identity was available (macro-F1 0.6344; weighted F1 0.9204). The failure should not be described as a small-sample null result.

This does not eliminate coverage limitations. The available manuscripts are heterogeneous in language, annotation practice, genre and size, and 37 candidate endpoints returned 404. Those limitations constrain class-level generalisation but do not explain away the locked failure.

### 2.2 Gross acquisition or parser failure — not supported

The corrected acquisition parsed all 29 downloaded manuscripts and recorded zero parse failures. Two TEI files (`gr1.recipes.xml` and `wo10.recipes.xml`) contained duplicate XML ID declarations. Recovery-mode parsing was introduced under a documented execution-only erratum. It did not alter the field ontology, role precedence, folds, features, estimators, thresholds or gates.

The recovered files contributed normally to the grouped folds. There is no evidence that the verdict is an artefact of wholesale parser loss. Annotation correctness at the semantic level has not been independently hand-audited, so local extraction or annotation errors remain possible.

### 2.3 Model and representation limitation — materially supported

Literal character n-grams recovered several roles well across manuscripts: INGREDIENT 0.7756 F1, TOOL 0.7277, OUTPUT 0.5964 and ACTION 0.5752. Identity-neutral representations were much weaker. The unsmoothed structural classifier reached macro-F1 0.3689, but the preregistered HMM/Viterbi stage reduced macro-F1 to 0.3270 by over-concentrating probability on common sequential states. Rare roles were especially damaged: structural-HMM F1 was 0.0072 for TOOL and 0.0111 for TEMPORAL.

This is a post-result diagnostic, not a revised gate. It shows that the frozen transition model was not an unqualified improvement despite the strong role-order signal. A different sequence model or hierarchical manuscript-aware representation might perform better, but such a model would require a new calibration version and a new locked test.

### 2.4 Genuine absence of recoverable procedural structure — rejected in the strong form

The corpus contains substantial recoverable procedural order. A first-order Markov model improved over IID by 0.7572 bits/token on average, and real ordering beat within-recipe shuffling by 0.9268 bits/token. Therefore the failure is not evidence that medieval recipes lack sequential operational organisation.

What failed was narrower: under manuscript holdout, the frozen identity-neutral surface representation plus estimator did not recover the seven token roles at the required level.

### 2.5 Irreducible non-identifiability — not established

The result does not determine whether operational roles are intrinsically unrecoverable without lexical identity. The lexical gate missed by 0.0279 and the identity-neutral HMM gate by 0.0230. These near-threshold gaps, together with the degradation caused by HMM smoothing, leave model limitation and annotation heterogeneity as live explanations.

No claim of intrinsic impossibility is warranted.

## 3. Scientific conclusion

CoReMA supplies a valid real medieval procedural control, but this implementation does not license semantic-role transfer to an unknown manuscript. The positive sequence result establishes organised procedure, not recoverable field semantics. The correct evidential state is:

- real historical procedural order: supported;
- manuscript-general lexical role recovery at the frozen standard: not supported;
- manuscript-general identity-neutral role recovery at the frozen standard: not supported;
- Voynich operational-role interpretation: not authorised.

This outcome strengthens the recoverability-first framework. Recognition of a historical domain, detection of orderly latent structure and recovery of operational variables are distinct achievements.
