# Voynich Frontier Programme v0.1 — Preregistration

**Status:** DRAFT FREEZE CANDIDATE  
**No new Voynich target result may be opened from code developed after this document until the protocol hash is recorded.**

## 1. Governing principles

1. Physical bifolia are the primary independent units whenever physical grouping is available.
2. Same-folio or same-bifolium leakage across train/test is prohibited.
3. Model selection, feature selection, thresholds and multiplicity rules are fixed before target opening.
4. Calibration must establish both power and false-positive control.
5. A failed instrument yields abstention, not evidence for the null mechanism.
6. Descriptive labels such as Currier A/B, Davis hand and section are not used to select an unsupervised latent-regime model.
7. Transliteration is part of the measurement model.
8. No post-target architecture rescue.
9. Every analytical claim receives a falsification criterion and a 60-day survival probability.
10. New findings are quarantined until an independent re-derivation or cold adversarial review.

## 2. Gate 0

### Required inputs

- canonical enriched records;
- canonical multi-transliterator slim container;
- P70-C specification if U4 is enabled;
- inherited physical-bifolium fold manifest;
- component-retention event inventory for the U1 adapter;
- any module-specific historical/visual control manifests.

### Pass

All required files exist and hashes are recorded; canonical counts and fields pass; physical units are not split; section names are exact; line order is internally coherent; all module seeds and target-opening flags default to sealed.

### Fail

Any required source is missing, silently reconstructed, has an unexplained hash mismatch, or the inherited fold manifest cannot be verified.

## 3. U1 — Transliteration uncertainty propagation

### H-U1

Load-bearing structural effects are not artifacts of one fixed transliterator's glyph-equivalence or word-boundary decisions.

### Uncertainty model

Primary empirical support is defined at the line level. One representative reading is taken from each independent transliterator family. Duplicate historical/version variants from the same family do not receive independent votes.

For each aligned line, preserve each complete observed reading. This intentionally keeps correlated glyph and boundary decisions together. Posterior sampling draws a complete observed family reading for a line rather than independently recombining uncertain characters into unattested strings.

Two complementary summaries are mandatory:

1. equal-family empirical posterior draws;
2. adversarial observed-reading envelope across all admitted families.

### Calibration

Before target-effect interpretation, synthetic corruption experiments must show:

- nominal 90% intervals cover the known ground-truth metric in at least 85% of calibration trials;
- sign-robustness calls have <=5% false-positive rate under zero-effect controls;
- planted effects at the declared minimum effect size are detected in >=80% of trials.

If this fails after the single permitted bounded repair, U1 returns `ABSTAIN_UNRESOLVED`.

### Primary robustness rule

A pre-existing effect is promoted to `MEASUREMENT_ROBUST` only when:

- posterior probability of the preregistered sign is >=0.975;
- the same sign holds in >=4/5 physical outer folds;
- the effect remains on the same side of the null under the admitted observed-reading envelope, or any envelope crossing is explicitly classified as representation-sensitive;
- no single transliterator family contributes >50% of the positive posterior mass in leave-one-family-out analysis.

No semantic interpretation is permitted.

## 4. U2 — D'Imperio 1978 replication

### H-U2

The five historically unstable Herbal-B pages are genuine within-corpus anomalies rather than artifacts of short samples and 1978 clustering.

### Mapping gate

Score the two surviving linear page-to-folio mappings against all available labelled Herbal samples. A mapping is admitted only if it reproduces >=26/28 declared labels. If neither does, reject the linear mapping and stop U2.

### Analysis

Use the same 40-page labelled panel and first 350–400 characters per page. Primary representation is monographic frequency, with three frozen unsupervised/leave-one-out anomaly instruments:

- correlation-distance agglomeration;
- nearest class-centroid correlation using leave-one-out centroids;
- robust within-class Mahalanobis/outlier score after shrinkage.

An historical anomaly is counted as replicated when at least two of three instruments flag it as a misassignment or within-class outlier under frozen thresholds.

Mandatory sensitivity:

- full-page length;
- 350-char and 400-char truncations;
- naive first/last character partition;
- each U1-admitted transliterator family.

### Verdict

Inherited unchanged:

- >=4/5: `CONFIRM_REPLICATION`
- <=1/5: `FALSIFY_REPLICATION`
- 2–3/5: `ABSTAIN_UNRESOLVED`

If anomalies appear only at short sample lengths and disappear at full-page length, classify them as sampling artifacts even if the historical replication threshold is met.

## 5. U3 — physical-unit latent regimes

### H-U3a

A small number of latent physical-unit regimes jointly explain multiple independent anomaly families.

### H-U3b

Different anomaly families require different latent regimes; there is no single common latent state.

### Unit and feature registry

Primary unit: complete physical bifolium.

Every feature is registered with:

- feature family;
- source artifact;
- estimator;
- whether it uses token identity;
- whether it depends on transliteration;
- whether it depends on layout/image features;
- nuisance covariates;
- missingness mechanism.

No feature may be created after Currier/hand/section associations are opened.

Equalise total weight by feature family before modelling.

### Models

Compare:

- K=1 null;
- Gaussian mixtures K=2..6;
- continuous low-rank factor model;
- feature-family-specific mixtures as a non-shared alternative.

Primary selection uses grouped held-out log likelihood plus bootstrap stability. BIC/silhouette are descriptive only.

### Calibration

Synthetic panels must include:

- one true regime with correlated nuisance;
- two and three shared regimes;
- independent regimes by feature family;
- continuous drift with no discrete states;
- section/hand mixtures without an additional latent state.

Required:

- correct broad model class in >=80% of synthetic trials at the declared effect floor;
- <=5% false discrete-regime calls under one-state and continuous-drift nulls.

### Target decision

A shared K>1 regime is admitted only if:

- held-out likelihood improves over K=1 in >=4/5 outer folds;
- stability ARI >=0.70 over the preregistered bootstrap;
- leave-one-feature-family-out solutions retain AMI >=0.60 to the primary solution;
- no single feature family carries >50% of total between-regime separation.

Only then reveal Currier, Davis hand, section and codicology.

## 6. U4 — surface closure and payload capacity

U4 inherits `NEXT_PROGRAMME_SURFACE_CLOSURE_AND_PAYLOAD_CAPACITY.md` except where this umbrella protocol is stricter.

### Additional U1/U3 gate

U4 may proceed only using effects classified as measurement-robust under U1, or explicitly labelled representation-conditional.

If U3 identifies stable regimes, U4 must either:

- model them as a frozen partially pooled covariate; or
- demonstrate that the surface model closes in each major regime separately.

No free regime-specific grammar is allowed.

### Surface closure

Payload testing remains sealed unless both predictive and forward-generative closure pass.

### Payload calibration

Carrier classes:

- multinary exact-variant choices;
- innovation/copy/mutation source choices;
- coordinated family choices;
- line-level state summaries;
- admitted subword-slot choices;
- sparse nomenclator-like events with changing line-local assignments.

Message classes:

- compressible Markov;
- naturalistic line-level class sequences;
- sparse bursts;
- changing-key/regime-specific mappings;
- iid random bits as an impossibility control.

Final rate grid is frozen before target opening from feasibility runs and must include a low-rate frontier.

A carrier is eligible only with >=80% power and <=5% false positives at its declared frontier.

### Target residual

Positive residual requires all of:

- positive total prequential MDL after full costs;
- >=4/5 folds;
- >=2 independent hand/section regimes;
- survives matched pure-surface simulation;
- survives exclusion of Scribe 3/Stars;
- lies above its calibrated rate frontier;
- multiplicity-controlled familywise decision.

## 7. U5 — verbose cipher

No Voynich score until fresh hidden-key examples can be both recovered and recognised.

Minimum target-opening gate:

- mean normalized plaintext recovery >=0.85;
- >=16/20 fresh-key trials with recovery >=0.75;
- family-recognition operational recall >=0.80 at precision >=0.95;
- matched-null false-positive rate <=0.05;
- source-family-disjoint test passed.

If recognition fails while recovery passes, report `RECOVERABLE_NOT_IDENTIFIABLE` and keep Voynich sealed.

## 8. U6 — VTPS v0.2

v0.1 is not a negative result. It failed instrument calibration.

A v0.2 visual instrument may open the target only if external known-writer/known-state controls achieve:

- nuisance-adjusted AUC >=0.80 on held-out manuscripts;
- >=80% detection at the frozen physical-effect floor;
- <=5% false positives for every declared nuisance-only null;
- robustness across normalized-ink and binary/skeleton views;
- document-level, never crop-random, holdout.

No Voynich retention label may be used to tune the representation.

## 9. Multiplicity

Each module has one primary familywise decision. Secondary metrics are descriptive unless explicitly registered.

Within modules, Holm correction is the default for a finite set of confirmatory p-values. Posterior-effect decisions use the frozen posterior threshold and do not additionally convert posterior probabilities to p-values.

Cross-module evidence is not pooled into an informal score.

## 10. Stopping rules

Stop a module when:

- calibration fails and the single repair is exhausted;
- target-opening prerequisites are unmet;
- source identity or mapping cannot be resolved;
- the result depends on post-target feature engineering;
- a required control has <=80% planned power at the admissible sample size;
- an implementation-equivalence test fails.

Partial outputs are retained and marked `ABSTAIN_UNRESOLVED`, not deleted.

## 11. Formal closeout vocabulary

Every closeout contains:

- `formal_verdict`
- `scope`
- `what_changed`
- `what_did_not_change`
- `calibration_status`
- `target_opened`
- `effect_or_frontier`
- `sensitivity`
- `retractions`
- `P_survives_60d`
- `next_allowed_action`
