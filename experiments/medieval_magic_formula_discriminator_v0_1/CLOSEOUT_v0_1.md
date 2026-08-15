# Medieval Magic Formula Discriminator v0.1 — Closeout

## Decision

**FAIL_EXTERNAL_Q2_Q4_STOP_BEFORE_VOYNICH**

Voynich remained sealed throughout. No RF, STA-family, full-STA, or connected-aaa result was computed.

## External qualification

| Gate | Result | Key statistic |
|---|---|---|
| Q1 synthetic recovery | PASS | macro-AUC 0.9626; accuracy 0.8188; bootstrap 95% CI 0.7625–0.875 |
| Q2 real A/B/C discrimination | FAIL | macro-AUC 0.6785 < 0.75; B AUC 0.5845 < 0.65 |
| Q3 family-leakage safety | PASS | grouped/random macro-AUC ratio 1.0089 >= 0.90 |
| Q4 nuisance controls | FAIL | full−nuisance AUC gain 0.0097 < 0.05; matched macro-AUC 0.6228 < 0.70 |
| Q5 permutation calibration | NOT RUN | frozen protocol required early stop after Q2/Q4 failure |

## Interpretation

The implementation can recover deliberately generated historical-mechanism controls, so the assay is not simply inert. However, on genuine external material it cannot distinguish ordinary medieval language (A), corrupted/hybrid charm language (B), and productive voces/formula language (C) at the preregistered strength.

The key failure is Q4: the full feature battery improves over nuisance-only features by only 0.0097 macro-AUC, and discrimination falls to 0.6228 after length/character matching. The B class falls to AUC 0.4333 on the matched set. Therefore much of the apparent external separation is attributable to gross form such as token/character-length structure rather than a robust charm-specific signal.

Q3 passing is useful: failure is not explained by formula-family leakage. The grouped split performs slightly better than the random split.

## Scientific consequence

v0.1 is **not qualified to answer whether the Voynich text resembles medieval magical/medical formula production**. This is an assay failure, not evidence that Voynich does or does not arise from such mechanisms.

The f116v charm evidence remains logically separate. The programme deliberately prevented a plausible marginal-charm identification from contaminating interpretation of the main manuscript.

## Any future v0.2

A new programme must be preregistered before further Voynich testing. The most defensible changes are external-only: enlarge the B corpus substantially with attested variant/corruption families; add real document-order medical miscellanies as D; increase source-witness diversity; and replace gross token-length/alphabet features with more sequence-level family-generation, local mutation, recurrence, and position-conditioned measures. Thresholds from v0.1 must not be relaxed post hoc.
