# Decision log — production_localisation_v1

This file records protocol changes and interpretive decisions. Entries must state whether they occurred **before** or **after** viewing primary model outputs.

| UTC date | Stage | Decision | Rationale | Effect on preregistered analysis | Author |
|---|---|---|---|---|---|
| 2026-09-01 | pre-data | Initialise v1 with strict IT, strict DE and diagnostic contact-zone cohorts | Separate workshop localisation from transmissible source content | Defines primary experiment; no results seen | ChatGPT / user-directed |
| 2026-09-01 | pre-result / cohort verification | Preserve the parchment-only primary design despite strong regional scarcity/imbalance emerging during authority verification | The target manuscript is parchment and material was preregistered before cohort discovery. Southern-German/Alemannic 1400–1450 catalogues are yielding many paper manuscripts and relatively few all-parchment codices, while northern-Italian parchment controls are easier to enumerate. Relaxing material eligibility now would change the estimand after seeing cohort composition. | No change to v1. The imbalance is an explicit limitation/methodological result. If v1 is underpowered under its preregistered threshold, it will be reported as such. A future v2 may broaden material and model parchment/paper explicitly, but cannot replace v1. | ChatGPT |
| 2026-09-01 | pre-result / implementation audit | Correct `run_primary_model.py` so known workshop/manuscript-family groups remain in the same cross-validation fold | `PREREG.md` requires grouped folds, but the initial script skeleton used ordinary repeated stratified folds. That implementation would allow related manuscripts/workshop siblings to leak across train/test folds. The discrepancy was identified before any primary model was fitted or any Beinecke classification was produced. | Implementation correction only; preregistered analysis is unchanged. Repeated grouped out-of-fold predictions will replace ordinary repeated `cross_val_predict`. | ChatGPT |
| 2026-09-01 | pre-result / cohort verification | Keep externally discovered authority controls in a separate enrichment ledger from the frozen ManuComp r15 extraction | The r15 strict parchment cohort is too small for the preregistered power target. External catalogues are therefore used to enumerate additional eligible controls, but provenance must show whether each control originated in r15 or external enrichment. | No eligibility-rule change. External enrichment candidates use the same frozen date/geography/material rules and are deduplicated before cohort freeze. | ChatGPT |

## Rules

- Do not delete superseded decisions; append a new row.
- Any threshold, geography, feature-family or exclusion-rule change after primary outputs are seen is post-hoc.
- Factual catalogue corrections are allowed but must identify affected manuscript rows and source IDs.
