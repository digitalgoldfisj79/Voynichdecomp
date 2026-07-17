# FROZEN PROTOCOL — Blind stroke-level palaeography v1

**Freeze date:** 2026-07-17  
**Branch:** `experiment/blind-stroke-palaeography-v1-20260717-full`  
**Scientific status at freeze:** no new Voynich hand model has been fitted or selected.  
**Scope:** independent visual recovery, or calibrated abstention, of scribal structure in Beinecke MS 408.

## 1. Question and admissible outcomes

Primary question:

> After controlling glyph/visual-family identity, section, Currier variety, physical quire and bifolium, folio, layout, crop geometry, scan/background properties and vocabulary, can stable scribal structure be recovered from character-conditioned stroke and ductus evidence?

Admissible Phase-I verdicts are:

1. `DISCRETE_K`: one value of K in 2–10 passes every frozen gate;
2. `HIERARCHICAL_OR_CONTINUOUS`: flat clustering fails but a calibrated non-discrete model passes;
3. `ABSTAIN`: no representation/model passes the frozen evidence rule.

`K=5` is a fixed candidate only. It receives no preference, tie-break, prior weight or interpretive advantage.

## 2. Operational blinding

### 2.1 Prohibited in Phase I

The Davis hand map, derived columns, line split, hand names and hand-coloured plots must not be loaded, imported, joined, searched, displayed or passed to any Phase-I code or model. Any existing dataset column known to encode that map is dropped at ingestion before any other operation. Prohibited field-name patterns include case-insensitive `davis`, `scribe`, `hand`, and known legacy encodings whose mapping has already been established.

The f115r target boundary is not supplied to feature extraction, model selection, threshold calibration or hyperparameter search. During Phase I it is identified only by a neutral reserved-unit ID and excluded from selection metrics.

### 2.2 Seal record

The adjudication map will be stored outside the Phase-I working tree. Before opening it, an independent utility records only its byte length and SHA-256 digest. The digest is permitted in the Phase-I freeze record; content is not.

### 2.3 Unsealing conditions

The map may be opened only after all of the following exist and are hashed:

- this protocol;
- data, feature, fold and model registries;
- exact selection and abstention code;
- external-control results and gate decision;
- all Phase-I K=2–10 outputs;
- selected model or formal abstention record;
- Phase-I medoids and uncertainty outputs.

No post-unseal model, feature, threshold, exclusion or tie-break modification can replace the frozen result.

## 3. Data units and exclusions

### 3.1 Voynich units

Retain all available forms of evidence without treating any segmentation as palaeographic truth:

- background-removed whole-word crops;
- transliteration-coordinate-conditioned windows used only for alignment/conditioning;
- connected components;
- small-gap merged components;
- multi-scale dense windows;
- skeleton and contour stroke fragments;
- line strips.

Connected components are proposals, not characters. EVA, v101, TT or other labels may condition homologous-form comparisons, but sequence statistics, word distributions and Davis-derived labels are not visual-hand evidence.

### 3.2 Independent units

Primary resampling unit: physical bifolium. No conjoint pages cross train/test boundaries. If a physical relation is unresolved, the entire quire is grouped conservatively. Primary validation is leave-one-quire-out where support permits; secondary validation is five deterministic physical-bifolium group folds stratified only by neutral metadata.

No crop-random, word-random or component-random train/test split is reportable.

### 3.3 Reserved material

- f115r is excluded from all model and threshold selection.
- Rose foldout mixed/collaborative material is excluded from the primary discrete-K fit and retained for post-selection descriptive assignment only.
- failed registrations and crops below frozen quality thresholds are excluded with reason codes, never silently.

## 4. Registration and quality gates

Inherited coordinate-transfer acceptance remains:

- at least 50 geometric inliers;
- inlier ratio at least 0.55;
- median reprojection error at most 3 pixels at the 2,500-pixel registration derivative;
- transformed source quadrilateral intersects the target canvas plausibly.

All candidate canvas scores and transforms are retained. Crop-level registration confidence is propagated into sensitivity analyses.

Image segmentation quality is audited on a deterministic sample stratified by folio, unit type and confidence. Formal corpus entry requires:

- at least 95% of sampled word boxes to overlap the intended text instance;
- at least 90% of sampled glyph-conditioned windows to contain the intended local form without truncating more than 10% of its visible ink;
- no more than 5% blank/background-only units in the primary visual-family pool.

Failure triggers repair of the coordinate/crop pipeline before external calibration or an explicit restriction to passing units; thresholds do not change.

## 5. Feature registry

All feature extraction is deterministic given the registry seed and model revisions.

### F0 — nuisance-only

No local stroke identity:

- width, height, aspect ratio, area and padding;
- page x/y coordinate, line position and relative line position;
- RGB/Lab brightness, contrast and colour statistics;
- background texture summaries;
- ink fraction, gross density and component count;
- word/transliteration length where alignment requires it;
- registration confidence, scan derivative, folio text density and illustration occupancy.

Categorical folio/quire identifiers may be used only to define groups or nuisance diagnostics, never as predictive inputs for external writer recovery or blind hand assignment.

### F1 — classical ductus

On normalized binary and greyscale ink views:

- skeleton endpoints, junctions, branches, cycles and geodesic-length distributions;
- contour direction and curvature histograms;
- Fourier contour descriptors;
- local slant and dominant orientation;
- distance-transform stroke-width distributions;
- terminal, crossing and loop geometry;
- run-length, connected-path and local binary-pattern texture descriptors.

### F2 — character/visual-family-conditioned residuals

Within every training fold:

1. define or fit the visual-family matcher using training units only;
2. estimate family centroids and family-specific nuisance regressions using training units only;
3. subtract expected family appearance and nuisance contribution;
4. transform held-out units with the frozen training-fold quantities.

Test samples never contribute to family centroids, PCA bases, regressions, whitening or normalization. Primary conditioning uses secure coordinate-aligned families. A secondary data-driven family graph is permitted only if learned on training folds and evaluated under shuffled-family nulls.

### F3 — dense DINOv3 correspondence

Frozen backbone: `facebook/dinov3-vitb16-pretrain-lvd1689m`, exact Hub revision recorded at execution. Use dense patch tokens, not pooled whole-word embeddings as the primary representation. Homologous patches are aligned by mutual nearest-neighbour correspondence within training-defined visual families. Features summarize residual patch displacement, local token differences and correspondence consistency.

DINO pooled embeddings remain audit/retrieval features only.

### F4 — historical HTR encoder

Primary historical encoder: `Riksarkivet/trocr-base-handwritten-hist-swe-2`, exact revision recorded. Use encoder hidden states pooled over foreground-aware spatial regions. Secondary: `microsoft/trocr-base-handwritten`. The primary comparison is the historical model; the generic model is an ablation.

### F5 — learned writer representation

A Siamese/metric encoder may be adapted on external known-writer corpora only. No Voynich hand target is used. The architecture is frozen after external calibration. Writer supervision is at writer ID; same-page pairs are prohibited as positive evaluation pairs.

### Ensemble

Candidate feature combinations are fixed before external results as:

- E1: F1 + F2;
- E2: F2 + F3;
- E3: F2 + F4;
- E4: F1 + F2 + F3 + F4;
- E5: externally trained F5 + F1.

Combination weights are selected solely on external development splits by nested group validation and frozen before Voynich Phase I. No Voynich-derived ensemble weighting is allowed.

## 6. Fold-local nuisance removal

For each outer training fold:

1. robust-scale each feature on training data;
2. fit cross-validated ridge regression from F0 to each palaeographic dimension on training data;
3. subtract fitted nuisance prediction from train and held-out data;
4. optionally remove a low-rank nuisance subspace chosen by inner training folds only;
5. whiten using training covariance with fixed shrinkage.

Report raw palaeographic, nuisance-only and residualized representations. A positive hand claim requires residualized evidence to pass; raw-only separation is insufficient.

## 7. External controls

### 7.1 Required corpora

Two primary external controls:

- Historical-WI: 3,600 pages, 720 writers, five pages per writer;
- HisFrag20: approximately 100,000 training and 20,000 test fragments with writer/page/fragment identifiers.

A medieval-script-focused secondary control is the Parzival database if access permits, because it contains a thirteenth-century Gothic manuscript written by three writers. Its result is supportive, not a substitute for the two primary controls.

### 7.2 External splits

- Historical-WI: leave-one-page-out retrieval, with no patches from the same page in the gallery for that query.
- HisFrag20: official train/test structure; all same-page fragments excluded from each query gallery.
- Hyperparameters and ensemble weights: nested writer-stratified, page-grouped development folds.

### 7.3 Metrics

Primary: mean average precision for writer retrieval with same-page exclusion. Secondary: top-1/top-5 writer retrieval, pairwise writer verification AUC, cluster adjusted Rand index on balanced known-K panels, exact-K recovery and calibration curves.

### 7.4 External confirmation gates

A representation/ensemble is eligible for Voynich only if all gates pass on both primary corpora:

1. writer-retrieval mAP exceeds nuisance-only by at least 0.05 absolute;
2. writer-retrieval mAP is at least 1.50 times nuisance-only mAP;
3. writer-label permutation test has empirical p <= 0.01 using at least 199 group-preserving permutations;
4. under crop jitter, scale, contrast, mild erosion and dilation, mAP retains at least 80% of its unperturbed value;
5. held-out character/visual-family or fragment-type mAP retains at least 70% of ordinary held-out-page mAP;
6. on matched balanced panels with true K in 2–10, exact K is recovered in at least 70% of panels and within ±1 in at least 90%;
7. on continuous-drift and nuisance-only simulations, false declaration of discrete K is at most 5%;
8. a no-writer-signal synthetic panel yields `ABSTAIN` in at least 95% of replications.

At least 40 independent calibration panels per mechanism are required. If any gate fails, Voynich Phase I remains unopened. One bounded pre-Voynich implementation repair is permitted only for a demonstrated code or optimization defect; it must be recorded and confirmed on fresh calibration seeds.

## 8. Phase-I blind models

### 8.1 Aggregation

Unit features are aggregated to line, folio and physical-bifolium representations with robust hierarchical means and uncertainty. Exact word types and visual families are balanced so frequent forms cannot dominate. Same-folio neighbours are excluded from retrieval evidence.

### 8.2 Flat candidates

Evaluate K=2–10 for each externally eligible representation using:

- diagonal-covariance Gaussian mixture with shrinkage;
- spherical mixture/von Mises–Fisher analogue for normalized features;
- spectral co-assignment clustering on a cross-bifolium similarity graph;
- Bayesian finite mixture with the same K range.

No single algorithm is decisive. The selected partition is a consensus co-assignment result whose rule is calibrated externally.

### 8.3 Continuous/hierarchical alternatives

Fit:

- hierarchical agglomerative tree with bootstrap edge support;
- probabilistic principal curve or low-dimensional style manifold;
- quire-aware Gaussian process/random-walk drift model;
- mixture-of-trajectories alternative.

These alternatives test whether apparent clusters are discretizations of continuous production drift.

### 8.4 Selection evidence

For every K and model report:

- held-out group predictive log likelihood;
- cross-bifolium writer-style retrieval consistency;
- bootstrap co-assignment stability and variation of information;
- recurrence across sections and Currier strata;
- held-out exact-word-type and held-out visual-family generalization;
- residual separation beyond F0 nuisance;
- segmentation and augmentation stability;
- behaviour relative to calibrated external model-selection distributions.

Silhouette score is descriptive only and never a selection gate.

### 8.5 Discrete-K confirmation rule

A K is eligible only if all conditions hold:

1. median physical-bifolium bootstrap ARI to the full-sample partition >= 0.70 and 10th percentile >= 0.50;
2. mean held-out group log likelihood exceeds the continuous one-cluster/nuisance baseline with paired bifolium-bootstrap probability >= 0.99;
3. cross-section recurrence: every retained cluster has at least two physical bifolia in at least two section/Currier strata, unless external calibration has prospectively identified a valid rare-cluster rule;
4. held-out exact-word-type and held-out visual-family performance each retain >= 70% of ordinary performance;
5. residualized evidence exceeds nuisance-only by >= 0.05 absolute retrieval consistency and >= 1.50 ratio;
6. the same K is selected in at least 70% of segmentation/augmentation perturbation runs and within ±1 in at least 90%;
7. no adversarial null attains an equal or better complete selection score at empirical p <= 0.01;
8. the continuous-drift model is not preferred by >= 10 expected log predictive density units per held-out bifolium on average.

If multiple K pass, select the smallest K within one standard error of the best externally calibrated composite score. This tie-break is independent of K=5.

If no K passes, flat result is `ABSTAIN` and the continuous/hierarchical result is adjudicated separately.

## 9. Nulls and adversarial controls

Required:

- section × Currier-preserving permutations;
- quire-preserving and mixed-quire permutations;
- folio/layout-matched nuisance simulations;
- exact-word-identity and transliteration-length controls;
- background-only representation;
- gross binary-ink-only representation;
- same-folio-neighbour exclusion;
- shuffled visual-family alignment;
- crop-geometry-only and page-coordinate-only models;
- continuous style drift with no true clusters;
- discrete mixtures differing only in background or crop generation;
- folio-ID prediction audit.

Every data-derived transform is refit inside each permuted training fold.

## 10. Stopping rule

The programme stops with the first formally valid state:

- external calibration fails after the one permitted bounded repair: `CALIBRATION_FAILURE`, Voynich unopened;
- external calibration passes and no Voynich model passes: `ABSTAIN`;
- a continuous/hierarchical structure passes while flat K fails: `HIERARCHICAL_OR_CONTINUOUS`;
- exactly one or tie-broken K passes: `DISCRETE_K`.

No additional feature family, threshold or model is added after Voynich Phase-I results are visible.

## 11. Davis adjudication after unseal

Without tuning compute:

- adjusted Rand index;
- adjusted mutual information;
- optimal one-to-one mapping accuracy;
- physical-bifolium bootstrap intervals;
- mapped cluster precision/recall and confusion;
- discriminating-folio results;
- hand-within-section and section-within-hand results;
- mixed-quire generalization;
- sensitivity to explicitly represented uncertainty in the expert partition.

Davis is an expert reference partition, not infallible ground truth.

## 12. Reserved f115r test

After model selection and unsealing, process all lines without supplying a boundary. For every possible between-line position report a calibrated change score/posterior. Report target-boundary rank, uncertainty, matched pseudo-boundary false-positive distribution and consistency of the two inferred segments with selected clusters. No target-boundary information may alter the model.

## 13. Reproducibility

- deterministic sorted inventories;
- explicit seeds and `PYTHONHASHSEED`;
- exact model revisions and package lock;
- atomic sharded writes;
- SHA-256 manifest for every code/config/data-index/result file;
- all failures, timeouts and amendments retained;
- clean-clone reproduction entry point;
- no destructive writes to prior repository, Supabase, Nomic or Hugging Face assets.

## 14. Freeze statement

This protocol fixes the scientific questions, external controls, feature families, nuisance treatment, folds, K range, model families, numeric gates, ensemble rule, nulls, tie-break, abstention rule, Davis adjudication and f115r test before any new blind Voynich model-selection result is computed.
