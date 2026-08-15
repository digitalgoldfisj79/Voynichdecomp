# Structured Production Specificity v0.3 — Completed Hostile Rerun

Date frozen: 2026-08-15
Status at freeze: prospective repair after v0.2 stopped at the representation-integrity gate. No v0.2 target F2/F3 metric values were produced before this protocol was frozen.

## Scientific question
Does the previously observed local F2/F3 resemblance between Voynich and productive magical formulae survive (a) removal of token-length metrics, (b) large real historical/non-magic control families, and (c) a within-Voynich matched random-operation null that preserves the local positional composition known from the broader programme to explain much apparent edit-1 structure?

This is a specificity/falsification test. Even a positive result would not establish magical content or mechanism.

## Hostile design constraints
1. No target-selected length metrics: `F2_tok_len_mean`, `F2_tok_len_std`, TTR, hapax, compression and entropy are excluded from the local-specificity decision.
2. `F2_shared_core_ratio` remains retired because its old implementation algebraically collapses to a token-length statistic. The repaired shared-core statistic stays only in the external candidate pool.
3. No synthetic control can qualify a metric. Primary qualification uses real text families only.
4. Corpus windows are measurement units, not source replications. External inference is across corpus families.
5. Target nulls operate within folio and consecutive five-line blocks, thereby fixing folio, section, Currier assignment, hand metadata, line count, token count, every token length and local within-word-position symbol inventories.
6. Representation layers are correlated sensitivity transformations, never independent replications and never majority votes.
7. f116v, imagery, palaeography and historical charm evidence are firewalled from this assay.
8. The external gate is rerun from zero. Voynich inputs are downloaded only after external PASS.

## Why v0.3 exists
v0.2 correctly stopped with `REPRESENTATION_INTEGRITY_FAIL` before scoring target metrics. Its AAA_CONNECTED mapping failed the coverage gate. The review also established that representation transforms were being overcounted as if independent.

The v0.3 repair is therefore pre-target and mechanical:
- `AAA_CONNECTED` is excluded from inference rather than repaired after seeing target scores.
- `RF_EVA` is the **primary measurement representation** because it preserves explicit surface word boundaries and uses observed Eva surface symbols, making its token granularity the closest available match to character-tokenized real-language controls.
- `STA_CODE` and `STA_FAMILY` are **sensitivity representations only**. They can force conflict if strongly opposite, but cannot create a positive result by voting.
- The target integrity gate requires exact locus/word-count parity across RF_EVA, STA_CODE and STA_FAMILY. Their dependency is explicitly measured and reported.

## External controls and magic comparator
Identical to frozen v0.2:
- real controls: LATIN_LLCT, LATIN_PROIEL, LATIN_ITTB, OLD_ITALIAN, GERMAN_GSD, FINNISH_TDT, TURKISH_BOUN, CREMMA_MEDICAL, CREMMA_ECCL, CREMMA_SCHOLASTIC, BVGS_RECIPE;
- comparator C: frozen Lecouteux productive-formula corpus, split by `source_work`.

Train/test partitions, block sizes (40, 80), family bootstrap, effect threshold and leave-one-family-out qualification are unchanged from v0.2. This prevents a fresh metric search after the representation failure.

## Candidate metric pool
Unchanged from v0.2 external qualification:
- F2_oneedit_degree
- F2_oneedit_component_frac
- F2_prefix_share
- F2_suffix_share
- F2_shared_core_frac_REPAIRED
- F2_branch_entropy
- F3_repeat_frac
- F3_adjacent_similarity
- F3_nearcopy_lag10
- F3_mutation_advantage
- F3_template_recurrence

Target scoring stops unless at least two metrics qualify externally.

## Primary target null
`N_LOCAL_COLUMNS_W5`, 250 surrogates per folio per representation:
- partition each folio into consecutive five-line blocks;
- preserve every token slot and length exactly;
- independently permute the j-th symbol among tokens in that block that possess a j-th position;
- preserve each block's symbol histogram at every within-word position;
- destroy cross-position/token-family coupling.

This is the direct matched null demanded by the hostile review and is deliberately stronger than token-order shuffling for type-level F2 metrics.

Secondary diagnostic `N_PREFIX_SUFFIX_W5` is unchanged from v0.2 and cannot rescue a failed primary result.

## Target metric inference
For each externally qualified metric m and each folio:
`oriented residual = external_C_direction * (observed_m - median(W5-null_m))`.

For each representation, bootstrap the median folio residual for each metric with 10,000 folio resamples.
Metric states:
- POSITIVE: CI lower bound > 0;
- NEGATIVE: CI upper bound < 0;
- NONRESOLVING otherwise.

A representation is descriptively `C_POSITIVE` only if:
- at least 4 qualified metrics are POSITIVE,
- zero qualified metrics are NEGATIVE,
- at least one F2 and one F3 metric are POSITIVE.

It is `C_NEGATIVE` only if at least 4 qualified metrics are NEGATIVE and zero are POSITIVE. Otherwise it is `METRIC_CONFLICT_OR_NONRESOLVING`.

These thresholds are frozen before v0.3 target scoring. They replace v0.2's excessively brittle requirement that every qualified metric share one sign.

## Representation decision rule
`RF_EVA` is primary. `STA_CODE` and `STA_FAMILY` are correlated sensitivity checks.

- `LOCAL_C_DIRECTION_SURVIVES_HOSTILE_NULLS`: RF_EVA is C_POSITIVE and no valid sensitivity representation is C_NEGATIVE.
- `LOCAL_SIGNAL_EXPLAINED_BY_MATCHED_NULL`: RF_EVA is C_NEGATIVE and no valid sensitivity representation is C_POSITIVE.
- `REPRESENTATION_CONFLICT`: primary and any valid sensitivity representation are confidently opposite.
- `LOCAL_SPECIFICITY_NONRESOLVING`: all other valid outcomes.
- `PRIMARY_REPRESENTATION_RESOLUTION_FAIL`: RF_EVA fails the pre-frozen resolution gate.

No majority vote is permitted.

## Resolution gate
Unchanged from v0.2:
- >=150 scored folios;
- >=max(40, 20% of folios) distinct observed composite scores at 12 decimals;
- largest exact-tie mass <=10%;
- median primary-null composite SD >1e-6.

## Stratified diagnostics
Section, Currier and hand summaries are descriptive only. Because the primary null is within folio, these labels are automatically held fixed; no stratum may rescue a failed whole-target decision.

## Stop conditions
- external infrastructure or adequacy fail: no target access;
- fewer than two qualified metrics: no target access;
- target word-boundary integrity fail: no target scoring;
- no target-side metric, null, threshold or representation change after scoring begins.

## Prior related result incorporated but not pooled
The separately frozen historical abbreviation operator programme returned `HISTORICAL_ABBREVIATION_NOT_SUPPORTED` on paired Nuremberg and ORIFLAMMS data and did not access Voynich. It is a distinct negative falsifier and is not pooled into the v0.3 statistic.
