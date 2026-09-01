# PREREGISTRATION — production_localisation_v1

Frozen before cohort extraction and before fitting any localisation model.

## 1. Primary research question

Can the **manufacturing/codicological profile** of Beinecke MS 408 discriminate between:

- **H_IT:** manufacture in a strict northern-Italian reference population; and
- **H_DE:** manufacture in a strict southern-German/Alemannic reference population;

with an explicit **H_X** outcome when Beinecke 408 is not well represented by either population.

This is not a test of where the manuscript's botanical, astronomical, zodiacal, medical, technical, or cryptographic source material originated.

## 2. Date window

Primary reference window: **1400–1450 CE**.

Primary manuscript eligibility requires an accepted date/date-range with midpoint in 1400–1450 and total stated uncertainty no wider than 50 years. Manuscripts dated only generically to “15th century” are excluded from the primary cohort.

Sensitivity window, declared now but analysed only after the primary result: **1380–1470 CE**.

## 3. Geography

### Strict northern-Italian cohort (IT)

Production explicitly localised by an authoritative catalogue or specialist study to one of:

- Lombardy
- Veneto
- Emilia-Romagna
- Piedmont
- Liguria
- Friuli-Venezia Giulia

Padua, Venice, Verona, Vicenza, Milan, Pavia, Ferrara, Bologna and other named places inside the above regions are eligible.

**Excluded from the strict IT training cohort:** Tuscany, Umbria, Marche, Lazio, Campania, Siena, and manuscripts described only as “Italy” or “northern Italy?” without firmer localisation.

### Strict southern-German/Alemannic cohort (DE)

Production explicitly localised to:

- Bavaria
- Swabia
- Franconia south of the Main where catalogued as southern German
- Alemannic German areas
- named southern-German centres such as Augsburg, Nuremberg, Regensburg, Munich, Ulm, Würzburg, Bamberg, Eichstätt, Konstanz where the catalogue explicitly localises production there

### Contact-zone diagnostic cohort (X; not used to train the primary binary model)

Tyrol/South Tyrol, Salzburg, Austrian border regions, Swiss/Alemannic areas whose classification is ambiguous, Alsace, Trentino, and manuscripts explicitly described as transalpine/mixed/uncertain between the two strict populations.

The X cohort is used only after the primary model is frozen, to evaluate whether Beinecke 408 resembles a contact-zone population.

### Localisation evidence rule

Current holding institution is never localisation evidence. Eligible evidence is an explicit place/region of production in an authoritative manuscript catalogue, codicological description, colophon, or specialist study. Stylistic attribution alone is retained as a secondary evidence field and does not satisfy primary eligibility by itself unless accepted by the catalogue as the production localisation.

## 4. Material and manuscript type

Primary cohorts:

- parchment manuscripts only;
- cohesive codices or production units whose relevant codicological features belong to one production event;
- no later composite volumes unless the original unit can be isolated confidently.

No subject-matter restriction in the primary cohort.

Secondary matched analysis (declared now): illustrated scientific, medical, technical, calendrical, astrological, or natural-history manuscripts only. This secondary analysis cannot replace the primary result.

## 5. Candidate universe and sampling

Candidate manuscripts will be enumerated from ManuComp plus authoritative external catalogues needed to validate metadata.

Every candidate considered must appear in `cohorts/eligibility.csv` with inclusion/exclusion status and reason.

Target: all eligible manuscripts up to **150 per strict cohort**. If more than 150 are eligible in a cohort, select 150 using deterministic stratified random sampling with seed `4081425`, stratifying by 10-year date bin and region so that no single centre dominates. If fewer than 75 eligible manuscripts exist in either strict cohort, the primary classifier is considered underpowered and results will be reported as exploratory.

Known direct copies, sister manuscripts, or multiple codices from the same tightly defined workshop campaign are grouped where identifiable. No group may contribute more than 5 manuscripts to the primary sample without a sensitivity analysis that downweights or holds out that group.

## 6. Primary feature families

Only the following families may enter the preregistered primary model. Exact coding is defined in `DATA_DICTIONARY.md`.

### A. Collation / gathering construction
- dominant bifolia per gathering
- proportion of quaternions / quinions / senions where recoverable
- variation in gathering size
- use of singletons or inserted leaves
- quire signatures / catchwords

### B. Page preparation and geometry
- parchment/paper confirmed
- pricking visible / absent / unknown
- ruling visible / absent / unknown
- ruling medium
- ruling pattern / text columns
- page aspect ratio
- text-block aspect ratio
- inner/outer/top/bottom margin ratios where measurable
- treatment of parchment flaws/holes
- hair/flesh arrangement where documented

### C. Production sequence
- text-first, image-first, preplanned/reserved-space, mixed, unknown
- text wraps around illustration
- reserved blank image spaces
- image/text collision or overlap behaviour
- underdrawing relationship where documented

### D. Scribal micro-practice independent of alphabet identity
- median line spacing normalised by page height
- baseline drift
- left-margin variance
- right-margin raggedness/compression
- paragraph indentation regime
- line-final compression frequency
- correction regime
- pen/nib width estimate where measurable

### E. Collaborative production structure
- number of scribal hands where securely identified
- hand changes sequential by page/gathering vs nested by bifolium vs mixed
- bifolium-level allocation indicator

## 7. Explicit exclusions from the primary localisation model

The primary model must not use:

- plant-image similarity
- zodiac/iconographic similarity
- architectural motifs, including merlons
- language/dialect of labels
- individual Voynich glyph shapes
- cipher/alphabet comparisons
- named textual works or source families
- pigments unless independently measured as a workshop/material feature with a separately preregistered protocol
- later provenance/ownership

These may be discussed after the primary result as contextual evidence only.

## 8. Missing data

Manual coding retains `unknown` rather than guessing. Features with >40% missingness across the combined strict cohorts are excluded from the primary fitted model before seeing Beinecke 408's classification result. Remaining missing values are imputed within cross-validation folds only:

- numeric: training-fold median
- categorical: explicit `unknown` level

A complete-case sensitivity analysis is required if at least 40 manuscripts per class remain.

## 9. Primary statistical model

Primary model: **L2-penalised logistic regression**, equal class weights, predictors standardised within each training fold.

Validation: repeated stratified 10-fold cross-validation, 20 repeats, seed `4081425`. Where a known workshop/manuscript-family group contains >1 sampled manuscript, all members of that group must stay in the same fold.

Model performance reported before classifying Beinecke 408:

- ROC-AUC
- balanced accuracy
- Brier score
- calibration slope/intercept
- confusion matrix at 0.5 threshold

If mean repeated-CV balanced accuracy is <0.65, the primary model is deemed insufficiently discriminating and Beinecke 408 will not receive a substantive IT/DE localisation claim.

## 10. Primary endpoint for Beinecke 408

Report the cross-validated model fitted to all frozen strict-cohort data as a descriptive **IT-vs-DE similarity probability**, not as an absolute historical posterior probability.

Decision labels:

- `IT-like`: P(IT) >= 0.70 and manuscript is not an outlier to the IT feature distribution
- `DE-like`: P(IT) <= 0.30 and manuscript is not an outlier to the DE feature distribution
- `indeterminate`: 0.30 < P(IT) < 0.70
- `out-of-distribution`: outside the 95% robust multivariate distance envelope of both strict cohorts

`H_X` is supported descriptively by an indeterminate or out-of-distribution result plus closer fit to the separately coded contact-zone cohort.

## 11. Secondary/sensitivity analyses (declared in advance)

1. Extended date window 1380–1470.
2. Subject-matched illustrated scientific/medical/technical subset.
3. Naive-Bayes categorical model as a low-complexity check.
4. Random-forest model only as exploratory non-linear sensitivity; never primary.
5. Leave-one-region-out checks.
6. Leave-one-major-workshop/family-out checks where groups can be identified.
7. Feature-family ablation: A–E separately and leave-one-family-out.

## 12. Contact sheets and visual audit

Any feature coded from images must have a reproducible contact sheet manifest containing manuscript ID, shelfmark, folio/page, source URL/IIIF canvas, crop box, transformation parameters and SHA-256 where locally materialised. Contact sheets must be generated from the manifest, not manually assembled to support a conclusion. Negative and ambiguous examples remain included.

## 13. Change control

This file is frozen once committed. Corrections to factual mistakes require a new version (`v1.1`, etc.) plus an entry in `decisions/DECISION_LOG.md`; the original remains in Git history. No threshold, geography definition, feature family, exclusion rule or decision boundary may be changed after viewing the primary classification without being labelled post-hoc.
