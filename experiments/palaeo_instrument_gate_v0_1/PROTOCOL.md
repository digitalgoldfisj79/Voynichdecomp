# Palaeographic Instrument Gate v0.1

**Frozen:** 2026-08-08, before any new G0A/G0C metric is computed.

**Branch:** `experiment/palaeo-instrument-gate-v0.1-20260808`

## Purpose

Determine whether any machine-vision representation is scientifically admissible for cross-manuscript palaeographic localisation. This is an instrument-validation programme, not a provenance test. No Voynich-to-comparator affinity score is permitted in this run.

The programme separates four logically distinct conditions:

- **G0A shape competence:** can the representation discriminate isolated shape identity at least as well as a deliberately simple morphology baseline?
- **G0B nuisance/background robustness:** does the representation avoid manuscript/acquisition identity after foreground normalisation?
- **G0C digitisation-source leakage:** can acquisition source still be recovered from the exact masked representation proposed downstream?
- **G0D design identifiability:** can provenance class and digitisation source be sufficiently crossed in the intended comparator panel?

A failure is a failure of the instrument/design, not evidence for or against any historical provenance hypothesis.

## Binding prior results

### Stage-5 visual confound result

The sealed Alpine–Venetian Corridor Stage 5 run tested DINOv3 manuscript-identity classification under leave-one-source-page-out prediction before any target similarity. Its `inkmask_v1` representation retained macro OVR AUC 0.7905697030, above the frozen failure threshold of 0.70. Therefore **G0B is already failed by an independent pre-target result**. It is not rerun or retuned here.

### SAGHOG nuisance result

The later full v1.5.1/stage-0 audit exists despite the Stage-5 closeout's stale repository audit. It showed that calibration-scaled acquisition+ink nuisance features matched or exceeded the selected representation on writer-disjoint evaluation. This is treated as additional motivation, not as a substitute for G0A/G0C.

### Unverified historical claim

A stored session record reports that a 2026-08-01 DINOv2-small 72×64 shape test was 15.3 null-SD worse than an eight-number naive descriptor. The exact artifact has not been recovered from the connected repository/history and **that number is not accepted as verified evidence**. G0A is rerun prospectively below.

## G0A — minimum isolated-shape competence

This is deliberately a floor test, not a medieval-palaeography validation.

Dataset: `sklearn.datasets.load_digits`, fixed local dataset, 1,797 isolated handwritten digit shapes, ten labels. Images are transformed to black foreground on white background and resized to **72×64** before representation extraction.

Split: stratified 70/30 train/test, `random_state=20260808`. Identical split for all representations.

### Fixed learned representation

- backbone: `facebook/dinov2-small`
- immutable revision: `ed25f3a31f01632728cabb09d1542f84ab7b0056`
- no fine-tuning
- final normalized CLS token
- standard Hugging Face image processor

### Frozen eight-number naive descriptor

All quantities are computed from foreground weight on normalized coordinates x,y in [-1,1]:

1. foreground mass fraction;
2. x centroid;
3. y centroid;
4. x variance;
5. y variance;
6. xy covariance;
7. horizontal-reflection absolute error;
8. vertical-reflection absolute error.

Both representations use a train-only `StandardScaler` followed by multinomial logistic regression with `C=1`, `max_iter=5000`, `random_state=20260808`.

Primary metric: test balanced accuracy.

Uncertainty: 2,000 paired bootstrap resamples of the held-out test set, seed 20260808, on `accuracy_embed - accuracy_naive`.

**G0A PASS:** lower endpoint of the percentile 95% paired-bootstrap CI is >= 0.

**G0A FAIL:** upper endpoint of that CI is < 0.

Otherwise: **G0A INDETERMINATE**.

The gate is intentionally demanding: a representation that cannot match a trivial morphology descriptor on isolated shape identity is not licensed for cross-manuscript palaeographic attribution.

## G0B — foreground/background robustness

No new metric is permitted. The sealed Stage-5 `inkmask_v1` result is binding:

- manuscript-identity macro OVR AUC = 0.7905697030;
- frozen failure threshold = AUC > 0.70.

Therefore **G0B = FAIL** before this run begins.

This failure does not terminate G0A or the diagnostic G0C because those distinguish shape incompetence from residual source/acquisition leakage.

## G0C — source-leakage stress test

The ideal fixed-glyph source classifier requires identical glyph identity across sources. The current Stage-5 crop corpus is not a glyph-labelled corpus, so the present test can decisively demonstrate leakage but cannot certify absence of leakage. A null result is therefore `INDETERMINATE`, never `PASS`.

Use only sources that cross geography within the sealed Stage-5 manifest:

- **BSB**: corridor-core `Cod.icon.242`; Bavarian controls `Clm 14684` and `Clm 14622`.
- **DigiVat**: corridor-core `Vat.lat.4082`; Bavarian control `Pal.lat.1362 B`.

Source label is BSB versus DigiVat. Geography labels are never inputs.

Two domain-swapped evaluations are fixed:

1. train source classifier on corridor-core manuscripts, test on Bavarian controls;
2. train on Bavarian controls, test on corridor-core manuscripts.

This prevents a source classifier from succeeding merely because source is synonymous with geography in the evaluation split.

Exact preprocessing is inherited from Stage 5 `inkmask_v1`: background-divided grayscale, threshold 215, black foreground/white background, square white padding.

Representations:

1. frozen eight-number morphology descriptor above;
2. binary masked pixels resized to 32×32 and flattened;
3. DINOv2-small normalized CLS token at the immutable revision above.

Classifier: train-only StandardScaler + logistic regression, `C=1`, `class_weight=balanced`, `max_iter=5000`, `random_state=20260808`.

Metrics for each transfer direction and representation: ROC-AUC and balanced accuracy. Pooled prediction is evaluated only after each direction has been generated out-of-domain.

Permutation test: 999 source-label permutations separately within each training domain, rerunning the classifier and evaluating the opposite domain; seed 20260808. The pooled observed statistic is mean directional AUC.

**G0C FAIL:** for any representation, mean directional AUC >= 0.70, both directional AUCs >= 0.65, and permutation p <= 0.05.

**G0C INDETERMINATE:** all other outcomes, because glyph identity is not held fixed.

There is no G0C PASS in this corpus. A future PASS requires a genuinely fixed-glyph or homologous-form multi-source control with an equivalence interval around chance.

## G0D — identifiability audit

No image metric is used.

A proposed provenance comparison is structurally admissible only when acquisition source is crossed with provenance class rather than nested inside it.

Minimum prospective criterion for a two-class comparison:

- each provenance class represented by >=2 independent digitisation sources;
- >=2 digitisation sources each contain manuscripts from both provenance classes;
- no single source contributes >60% of either class;
- VMS/Beinecke may remain a single out-of-domain target source, but it cannot be used to tune the representation or classifier.

G0D is evaluated separately for:

- Padua/Veneto vs German/Bavarian;
- Padua/Veneto vs Lombardy/Pavia.

`PASS` requires all minimum criteria. `REPAIRABLE` means known same-source controls make the crossing feasible but the panel is not yet assembled. `FAIL` means the available target design is structurally nested and no known repair has been identified.

## Stopping rule

No cross-manuscript Voynich localisation score may be computed unless a future instrument has:

1. G0A PASS;
2. G0B PASS on an independent corpus;
3. G0C PASS on a fixed-glyph/homologous-form multi-source corpus;
4. G0D PASS for the historical comparison being claimed.

This run may continue through diagnostic gates after a failure solely to identify the failure mode. It cannot unlock provenance inference.
