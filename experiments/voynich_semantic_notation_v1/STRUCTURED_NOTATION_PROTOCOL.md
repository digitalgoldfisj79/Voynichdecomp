# Voynich Structured Notation Programme — VSN-v1

Date frozen: 2026-08-12 (Europe/London)
Branch: `experiment/voynich-semantic-notation-v1-20260812`
Namespace: `VSN-v1`

## 1. Scientific question

Primary question: do controlled written-component changes in Voynichese predict independently observable manuscript content after page/quire, section, layout, frequency and local token-family structure are controlled?

This programme does **not** test a Bavarian/German homophonic-transducer hypothesis and must not be interpreted as a plaintext-language identification exercise.

Competing interpretations:

- H_SN: at least some recurring written components behave as reusable category/property/operator/state features.
- H_PS: surface morphology is structurally sophisticated but semantically empty or only weakly semantic generated pseudo-text.

A positive result requires held-out predictive transfer. Raw association is not evidence for H_SN.

## 2. Representation freeze

Primary representation: frozen RF / Basic EVA, `voynich_nu_data_RF` in Supabase, corresponding to the existing frozen RF source infrastructure used by the prior STA programme.

Primary-token eligibility: exact lowercase `[a-z]+` RF tokens only. Tokens containing commas, uncertainty symbols or other non-letter marks remain in the occurrence ledger but are excluded from primary morphology inference. No normalization of ambiguity marks into letters is permitted.

STA-family, full STA and connected analytical AAA are replication representations only. They may be run only if RF discovery qualifies; they must reuse the already pinned conversion/source machinery in the repository rather than introduce a new transliteration mapping.

## 3. Occurrence dataset

Snapshot schema: `voynich_semantic_notation_v1` in Supabase.

Primary table: `rf_occurrences`, one row per dot-delimited RF token occurrence, preserving folio, quire, locus/line number, IVTFF layout marker, line position, paragraph-like block index, token frequency and available page metadata.

External visual table: `visual_targets`, one existing non-spurious Voynich whole-plant and root target per folio where a 3072-dimensional stored embedding exists.

Missingness is explicit. Word x/y coordinates and Davis/Currier hand fields are currently unavailable in the populated database and are **not imputed**. Their absence lowers the strength of causal control but does not invalidate page-level/quire-level tests. Any later restoration of those fields is a protocol amendment and cannot alter frozen confirmation data or thresholds.

## 4. Morphology discovery independent of images

Candidate morphology is generated without consulting visual outcomes.

### 4.1 RF edit graph

Construct token types and all Levenshtein-distance-1 minimal pairs using deletion signatures. Record edit position as prefix / suffix / internal and the changed component. This captures one-component insertion/deletion/substitution families without hard-coding named Voynich examples.

### 4.2 Affix inventory

Enumerate exact prefixes and suffixes of lengths 1–4. A candidate affix enters the discovery inventory only if:

- at least 50 eligible RF occurrences contain it in the relevant position;
- at least 20 distinct token types contain it;
- there are at least 10 attested contrast cores for which both a component-bearing form and a matched non-bearing or alternative-component form exist.

These support thresholds are fixed before visual testing.

### 4.3 Family identity

For a component contrast, `family/core` is the residual token after applying the frozen component edit. Cross-family transfer must train and test on disjoint residual cores; mere recurrence of the same full token is disallowed.

## 5. Discovery / confirmation split

The split unit is quire where quire metadata exist; folio is used only for records with missing quire.

Deterministic split key:

`SHA256("VSN-v1-confirmation::" + block_id)`

Blocks are sorted by hash. Approximately 25% of blocks, selected without reference to morphology or visual outcomes, are sealed CONFIRMATION. The remainder are DISCOVERY. A block is never split across arms.

Within DISCOVERY, model assessment is leave-one-quire-out (or leave-one-folio-block-out for unquired records).

No candidate fishing is permitted in CONFIRMATION. Before opening confirmation outcomes, freeze candidate component, contrast definition, direction/sign, external feature, model family and threshold.

## 6. Primary statistical estimand

For external target F and morphology component M compare:

BASE: F ~ context controls + family/core controls

FULL: F ~ context controls + family/core controls + M

Primary evidence is out-of-sample improvement FULL − BASE on held-out blocks. Depending on target type the score is log loss/deviance, squared-error reduction or cross-validated retrieval/metric score. Inference is clustered at folio/quire level; token occurrences on the same page are never treated as independent manuscripts.

Mandatory controls when available: quire/folio block, section, IVTFF layout family/marker, line position, token length, log token frequency, family/core identity. Hand and spatial word coordinates enter only if independently populated before a confirmation test is frozen.

## 7. Annotation-free visual arm

Initial qualified domain: herbal pages, because existing 3072-dimensional whole-plant/root embeddings permit testing without new subjective labels.

Primary tests:

1. residualised component-from-visual prediction;
2. conditional two-sample / distance test in embedding space;
3. cross-family operator transfer: train component effect on residual cores A, test on disjoint cores B;
4. leave-quire-out evaluation.

Dimensionality reduction, if required, is fitted on training folds only. No PCA/PLS/CCA basis may see held-out blocks.

## 8. Human-annotation arm

Existing page-level feature annotations may be used as a separate arm if their provenance is logged. Free-text page descriptions are not a primary endpoint. Human-coded features and embedding results must be reported separately.

## 9. Hostile controls

Required before confirmation:

1. leave-one-quire-out / block-out validation;
2. morphology permutation within section × layout-family × frequency-bin strata, preserving page counts;
3. spatial/page null: reassign visual targets among geometrically/contextually comparable pages within the same domain/quire stratum where possible;
4. synthetic structured-text null preserving token/component frequencies, morphology, section and line-position biases but breaking text-image linkage;
5. frequency-only BASE comparator;
6. multiple-testing correction across the discovery family (Benjamini–Hochberg FDR q <= 0.05), followed by a small frozen confirmation family.

## 10. Qualification gates

A candidate is eligible for confirmation only if all are true in DISCOVERY:

- support thresholds in section 4 are met;
- leave-block-out FULL beats BASE in the prespecified direction;
- 95% block-bootstrap interval for predictive improvement excludes 0;
- discovery-family BH-FDR q <= 0.05;
- cross-family transfer is positive on disjoint cores;
- the observed score exceeds the 99th percentile of the matched permutation/null distribution;
- no single quire contributes >50% of the aggregate gain;
- estimated power for the frozen confirmation test is >=0.80 at the discovery effect size after a 25% shrinkage penalty.

These gates deliberately privilege specificity over sensitivity. A classifier accuracy threshold alone cannot qualify a result.

## 11. Confirmation success

Confirmation succeeds only if the frozen candidate has the same sign/direction, FULL beats BASE on sealed blocks, and the prespecified confirmation p-value / empirical-null tail probability is <=0.05 after correction for the frozen confirmation family. Failure closes that candidate; no rescue variant under `VSN-v1`.

## 12. Historical workstream firewall

Workstream B may identify historical mechanisms but cannot alter Workstream A candidate generation, split, target definitions or thresholds. Workstream A behavioural primitives are frozen before mechanism matching.

Historical grades:

- A: exact structural precedent — extended compositional notation replaces ordinary prose for technical information.
- B: strong mechanism precedent — base sign + modifier/context systematically encodes multiple attributes.
- C: pedagogical precedent — curriculum/manuscript demonstrably trains formal multidimensional representation but does not itself meet A/B.
- D: generic abbreviation/symbolism only.
- E: irrelevant resemblance.

No glyph-shape resemblance is evidence.

## 13. Stopping rules

Stop rather than iterate variants if adequate-power morphology adds no held-out value; leave-quire-out removes the effect; embedding effects fail matched nulls; synthetic pseudo-text matches the result; frozen confirmation fails; or historical search yields only C/D generic mechanisms.

## 14. Contamination ledger

Known prior knowledge before target analysis includes common Voynich strings such as `qokeedy/qokedy/qokeey`, `qo-`, gallows and common suffix discussions. These are **not** privileged candidates. Candidate generation is exhaustive under section 4.

The programme may reuse prior RF/STA/AAA conversion code and existing image/mask/embedding assets. It may not reuse prior historical expectations to select morphology-feature pairs.

## 15. Compute discipline

Prefer existing embeddings and database assets. Any HF/Fal job must have an explicit timeout and job identifier; jobs must be cancelled when no longer useful. Programme closeout must record zero known running exploratory jobs.
