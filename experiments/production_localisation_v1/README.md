# Voynich production-localisation study v1

Status: **preregistered / data collection not yet started**

Primary question: **Does the physical manufacture of Beinecke MS 408 fit a securely localised northern-Italian manuscript population better than a securely localised southern-German/Alemannic population, or neither?**

This experiment deliberately separates **workshop/manufacturing evidence** from **source/content evidence**. Italian herbals, German zodiac imagery, technical exemplars, and other transmissible content are not used in the primary localisation model.

## Hypotheses

- **H_IT** — Beinecke 408 was manufactured in northern Italy and acquired German/Alemannic source material or personnel through transmission.
- **H_DE** — Beinecke 408 was manufactured in the southern-German/Alemannic sphere and acquired northern-Italian source material through transmission.
- **H_X** — Beinecke 408 is not well described by either strict reference population (mixed/contact-zone, atypical workshop, or insufficient signal).

## Principle

Localise the **workshop**, not the sources.

Primary variables are mundane production behaviours that are difficult to inherit from an exemplar: gathering structure, ruling/pricking, page geometry, production sequence, scribal allocation, line and margin habits, and other codicological/scribal micro-practices.

## Repository structure

- `PREREG.md` — frozen hypotheses, inclusion/exclusion rules, outcomes, stopping rules and statistical plan.
- `DATA_DICTIONARY.md` — exact feature definitions and coding rules.
- `cohorts/eligibility.csv` — every candidate manuscript considered, including exclusions and reasons.
- `cohorts/frozen_sample.csv` — immutable primary sample after eligibility and deterministic sampling.
- `data/workshop_features.csv` — one row per manuscript, source-backed feature values.
- `data/feature_sources.csv` — field-level provenance: catalogue/page/URL/quote or image reference supporting each coded value.
- `data/vms_features.csv` — Beinecke 408 coded under the same schema.
- `provenance/sources.csv` — bibliographic and catalogue source ledger.
- `provenance/assets.csv` — image/contact-sheet asset hashes, URLs, crop coordinates and licensing notes.
- `decisions/DECISION_LOG.md` — every protocol change, with date, rationale and whether it occurred before or after seeing results.
- `contact_sheets/` — generated, labelled visual audit sheets; never hand-selected without a manifest.
- `scripts/` — deterministic cohort freezing, validation and analysis code.
- `results/` — machine-readable outputs and report tables; no result is overwritten.

## Transparency rules

1. Current holding institution is never treated as production location.
2. A localisation value must point to an explicit catalogue/scholarly source; inference from style alone is coded separately.
3. Unknowns remain unknown; no silent imputation during manual coding.
4. Primary analysis uses only features frozen in `PREREG.md` before the comparison cohorts are inspected statistically.
5. All exclusions are retained in `eligibility.csv` with reasons.
6. Every manually coded feature has field-level provenance in `feature_sources.csv`.
7. Contact sheets include all items specified by their manifest, including negatives.
8. User-supplied or copyright-sensitive images are referenced by checksum and source description unless redistribution is explicitly permitted.
9. Scripts use fixed random seeds and write versioned outputs.
10. Any post-hoc analysis is labelled **exploratory** and never substituted for the preregistered primary result.

## Versioning

`v1` is intentionally conservative. It asks only whether workshop/manufacturing evidence can discriminate the two main production hypotheses independently of subject matter. Later palaeographic, material-science or content/source analyses must be separate modules and cannot retroactively alter the v1 primary endpoint.
