# P1 memory/recurrence source-identification v1 — implementation freeze

Date: 2026-09-07
Status: FROZEN BEFORE ANY P1 SOURCE-IDENTIFICATION OUTCOME.
Parent manifest SHA256: `62655854117793168d46bfb05b548d5993f28dd613301d859fd666504a3b51c0`.

This note resolves implementation details that were deliberately schematic in the already-frozen manifest. It does not change the five named sources, three unknown controls, 20 feature families, split sizes, classifier family, abstention thresholds, validation gates, representation gate, power grids, or target seal.

## Sequential recurrence features
For each held-out physical fold, events are grouped by page in canonical event order. Opportunities are pairs at positional distance d within the same page. Distance buckets are exactly 1, 2–5, 6–16, and 17–64.

Channels are exact-token equality, frozen family equality, and ordinary unit-cost ED1 as defined in the inherited Stage-2 implementation. For each channel/bucket, the observed pooled hit rate is compared with the mean pooled hit rate across 64 deterministic within-page token-order permutations. Permutations preserve each page token multiset, line-length/position skeleton and metadata. The same event-index permutations are used for R0 and R1.

Rates use Jeffreys smoothing `(hits + 0.5)/(opportunities + 1)`. Feature value is `log2(smoothed observed rate / smoothed 64-permutation mean rate)`.

## Boundary features
Physical adjacent pairs are partitioned into wholly-within-line pairs and pairs crossing from the final token of one physical line to the first token of the next line on the same page. Exact and ED1 excesses are each computed against the same 64 within-page permutations with line positions held fixed. Boundary feature = cross-line log2 excess minus wholly-within-line log2 excess.

## Page/repertoire features
- Adjacent-page exact/family Jaccard: mean Jaccard only for pages that are consecutive in the canonical full-manuscript page order and both occur in the held-out fold.
- Bifolium-minus-cross exact Jaccard: mean exact-token Jaccard over page pairs within the same held-out physical bifolium minus a local cross-bifolium control. For each held-out page, the control is the nearest page in canonical page rank belonging to another bifolium; ties break toward earlier canonical rank.
- Page new-type slope: exactly the inherited Stage-2 within-page first-occurrence hazard slope over five relative page-position bins; it is computed directly from the transformed held-out representation.

## Representations
R0 uses canonical token strings and frozen families. R1 collapses immediately repeated identical glyphs within every token, then recomputes the frozen first-glyph/last-glyph/length-bin family. All recurrence, boundary, Jaccard and new-type features use the representation-specific tokens/families. The two held-out predictive codelength gains remain those of the canonical fitted surface model and are identical inputs to R0/R1 classifiers.

## Hidden-source generation
The source mechanisms are the already-frozen T0b implementations. Synthetic train and test corpora are generated independently on the inherited train/test skeleton for each physical fold. The analysis model is then refit from synthetic training only and reselects regional lambda using the inherited training-only grid.

- `working_set_only`: RegionalModel with source lambda 0.
- `r64_exact`: RegionalModel with inherited source lambdas `.03,.03,.03,.02,.03` by fold.
- `refractory_exact`: T0b refractory generator, rho=.04.
- `line_reset_r64`: T0b physical-line-reset regional generator.
- `family_persistence`: T0b family-persistence generator, rho=.04.
- Unknown controls: T0b gross page shuffle, gross page×family shuffle, and line opener omega=.05.

Power curves vary only the already-prespecified source parameter: refractory rho, family-persistence rho, or R64 source lambda. No classifier/feature/threshold changes are permitted from power outcomes.

## Classifier and abstention
For each representation and physical fold, feature standardisation is learned on development trials only. One multinomial L2 logistic model is fit per physical fold. C is selected from `{0.1,1,10}` by five-fold grouped development CV, minimizing multiclass log loss; all records derived from one manuscript trial stay in one CV group.

For a manuscript trial, fold log probabilities are summed classwise and renormalised to a manuscript posterior. Prediction requires top posterior >=.70 and top1-top2 posterior margin >=.20.

OOD is manuscript-level. The five fold-standardised 20-vectors are concatenated to 100 dimensions. A class-specific Ledoit-Wolf covariance and centre are learned from the 80 development manuscripts of that class; the OOD threshold is the empirical 97.5th percentile of development shrinkage-Mahalanobis distance. The predicted class must fall inside its class envelope or the instrument abstains. Continuous OOD score is predicted-class distance divided by that class threshold.

## Pairwise separation
For each unordered source pair A/B, use only untouched validation manuscripts truly generated by A or B. Trial margin is `log P(true class) - log P(other class)`. The observed statistic is mean margin. The matched null permutes the A/B labels across the complete manuscript-trial probability rows 999 times while preserving 50/50 class counts. Report `(observed - null mean)/null SD`; the qualification gate requires absolute value >=2 for every source pair.

## Unknown controls
An unknown control is rejected when the frozen classifier abstains. Gate requires >=80% rejection for every unknown control. The continuous OOD separation is `(unknown mean OOD - known-source validation mean OOD) / known-source validation OOD SD`; gate requires >=2 for every control.

## Target seal
No P1 runner contains a real-Voynich mode. Target scoring can only be added after both representations pass validation, pairwise separation and unknown-control gates and the frozen power curves are reported. A failed v1 closes v1 and requires a new version with fresh synthetic splits.
