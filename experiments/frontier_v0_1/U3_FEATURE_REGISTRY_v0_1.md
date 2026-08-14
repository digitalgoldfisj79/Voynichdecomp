# U3 feature registry v0.1

Date frozen: 2026-08-14
Status: **FROZEN BEFORE U3 TARGET FEATURE MATRIX IS CALCULATED**

This file completes the estimator registry required by the existing U3 preregistration. It does not alter the nine feature slots already frozen in `templates/U3_FEATURE_MATRIX_TEMPLATE.csv`, the K grid, folds, thresholds or target decision rule.

Primary unit is the complete physical bifolium from `data/fold_manifest.json`. Currier, hand and section are not used to construct, select or calibrate the latent model.

## Registered features

### `TEXT_ORDER::adjacent_mi`
Source: canonical `enriched_records.pkl`.
Estimator: line-internal adjacent-token mutual information using a *global* top-100 token vocabulary plus `OTHER`, with the same vocabulary for every bifolium. Report the plug-in MI minus the mean MI from 64 independent within-line token permutations. Seed is `20260814 + bifolium_ordinal`. Negative corrected values are retained. This is the historically used top-100 robustness frame with a matched random-operation null; no page/section labels enter it.
Uses token identity: YES. Transliteration dependent: YES. Layout/image dependent: NO.

### `TEXT_ENTROPY::red1`
Source: canonical `enriched_records.pkl` EVA token surface.
Estimator: concatenate each line with a single space symbol between tokens, never across line boundaries. Let H0=H(X_t), H1=H(X_t|X_{t-1}). `RED1=(H0-H1)/H0` by plug-in counts. Entropy order is always stated explicitly.
Uses token identity: only through characters. Transliteration dependent: YES.

### `TEXT_ENTROPY::red2`
Same stream and counts as RED1. Let H2=H(X_t|X_{t-2},X_{t-1}). `RED2=(H1-H2)/H1`.

### `TEXT_EDIT::ed1_density`
Source: canonical `enriched_records.pkl`.
Estimator: for each manuscript page inside the bifolium, calculate the occurrence-weighted probability that two distinct token occurrences drawn without replacement have *different token strings at Levenshtein distance exactly one*. Equivalently, numerator is the sum of `count(type_i)*count(type_j)` over unordered ED1 type pairs and denominator is `C(N,2)`. Bifolium value is the denominator-weighted mean over its pages. ED1 type pairs are enumerated exactly using deletion/substitution/insertion neighbourhood logic, not approximate embeddings.
Uses token identity: YES. Transliteration dependent: YES.

### `TEXT_PERSIST::midfix_lag1`
Source: frozen structural-persistence event table, SHA-256 `dec8708b380c7b85e40967240f21468c2a636ce5ac3e3761b9a4793cf3258eec`.
Estimator: inherited `structural_tournament.py` contract. Sort by folio/line/order/event. Current event is eligible iff immediately previous event has the same `page_key` and the same frozen broad `family`. Parse Stolfi prefix/midfix/suffix using the inherited EVA-unit parser. Among eligible adjacent pairs where both parses are normal and the previous midfix is nonempty, return the fraction with identical midfix. Exact whole-token repeats are not removed, matching the inherited observed statistic.

### `TEXT_PERSIST::suffix_lag1`
Identical eligibility and parser contract to midfix; among eligible pairs with nonempty previous suffix, fraction with identical suffix.

### `LEXICAL::hapax`
Source: canonical `enriched_records.pkl`.
Estimator: number of token types occurring exactly once in the bifolium divided by number of distinct token types in the bifolium.

### `LEXICAL::type_token`
Distinct token types divided by token occurrences in the bifolium.

### `PAGE::between_page_overlap`
Source: canonical `enriched_records.pkl` plus frozen bifolium membership.
Estimator: construct the token-type set for every distinct manuscript page represented in the bifolium. Return the arithmetic mean of pairwise Jaccard overlaps `|A∩B|/|A∪B|` across all page pairs. A bifolium with fewer than two represented pages is missing for this feature; no imputed target value is created by the builder.

## Nuisance and missingness policy

The feature builder records `n_tokens`, `n_chars`, `n_pages`, `midfix_n`, and `suffix_n` in a sidecar audit file, not in the latent-model matrix. The latent model's existing training-fold median imputation handles genuine missing features. No feature is residualised or selected after observing Currier/hand/section.

Because feature families contain unequal numbers of registered columns, the already-frozen U3 model gives each feature family equal aggregate weight by multiplying each column by `1/sqrt(number_of_columns_in_family)`.

## Calibration floor

Synthetic calibration is performed before any real feature matrix is passed to `latent_regime.py`. The declared discrete-regime effect floor is a **1.50 pooled-SD mean displacement distributed across at least three independent feature families**. This is a deliberately substantial latent state: U3 is not licensed to claim weak sub-threshold regimes. One-state, continuous-drift, nuisance-only, shared K=2/K=3, and family-specific-regime controls use N=50 and the inherited five-fold membership sizes.

Calibration must satisfy the already-frozen programme rule: ≥80% correct broad-class calls at the declared effect floor and ≤5% false discrete calls under one-state and continuous-drift nulls. Failure yields U3 `ABSTAIN_UNRESOLVED` without target preselection.
