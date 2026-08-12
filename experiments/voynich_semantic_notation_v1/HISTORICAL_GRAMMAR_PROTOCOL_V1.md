# VSN-v1 Workstream B2 — Historical Grammar Structural Comparison Protocol v1

Frozen: 2026-08-12 (Europe/London)
Branch: `experiment/voynich-semantic-notation-v1-20260812`
Namespace: `VSN-B2-v1`

## Scientific question

Do independently attested northern-Italian compositional information-encoding mechanisms from c.1420–1434 naturally produce structural properties resembling Voynich morphology **without tuning the historical mechanism to Voynich**?

This is a mechanism test, not a derivation claim and not a plaintext-decoding exercise.

## Source firewall

Historical grammars are extracted and frozen before new Voynich target metrics are queried for this experiment. Source rules may be taken only from the cited critical edition/manuscript scholarship. Where a source does not define a written serialization, none will be invented merely to make it comparable to Voynich strings.

The following are explicitly forbidden:

- free Cartesian recombination of historical slots unless the source licenses it;
- choosing component inventories to match Voynich frequencies;
- changing slot count after seeing Voynich metrics;
- mapping historical semantic values to particular EVA/STA glyphs;
- combining Matteo, Bartolomeo, Ragona and Vat. lat. 10488 into a synthetic hybrid grammar and presenting that hybrid as historically attested;
- treating spatial mnemonic fields as written token positions unless the source itself serializes them.

## Historical systems

### H1 — Matteo da Verona, `De arte memorandi` (Padua 1420/1423)

Attested operations:

1. **Composite artificial words**: selected source words are represented by their first syllables and concatenated in order. Explicit examples/rules support 2, 3 and 5 first-syllable compounds, and recursive higher-order compounding is explicitly contemplated.
2. **Medical quality/degree encoding**: four primary-quality pair classes; two side/dimension values within a class; ordinal degrees 1–4 encoded vertically.
3. **Typed logical/textual loci**: position can encode proposition class, book/chapter/question/article order and other typed information.

For H1, only the explicit concatenative artificial-word operation is eligible for surface-string comparison. The medical and logical systems are abstract structured-state comparators only.

### H2 — Bartolomeo da Mantova, `Liber memoriae artificialis` (Mantua 1429)

Attested architecture:

- 10 architectural loci;
- 10 locus-objects within each architectural locus;
- 3 associated images per locus-object, ordered by type;
- each four-item table contains one locus-object plus three images;
- one artificial four-syllable codeword is formed by concatenating the first syllables of the four names in order.

This yields 100 prepared four-item tables from 100 locus-objects + 300 associated images. Cacopardo's prose states `400 words -> twenty codewords`; this is arithmetically inconsistent with the immediately described 100-table architecture and is treated as a probable secondary-source numerical error. The experiment does **not** use `20` as a grammar parameter.

H2 does **not** license free recombination across the four slots. Exact surface-token comparison therefore requires transcription of the prepared codeword inventory. Until that inventory is recovered, H2 is compared only on frozen architecture-level descriptors.

### H3 — Jacopo Ragona, `Artificialis memoriae regulae` (1434)

Attested debt-record fields include day, month, year, debtor name, father, lineage, weight, measure, money amount and debit-versus-credit state. These are assigned to stable body locations/order. Weekdays have fixed metal mappings and months fixed image mappings. Ragona also syllabifies unknown names for image construction.

H3 is a typed spatial-record grammar. No linear written-token serialization is attested in the extracted passage, so no character-string generator is permitted.

### H4 — Vat. lat. 10488 (Venice 1424)

Attested operational notation includes coefficients plus abbreviated algebraic species, superscript/spatial placement, formal fractions and several variants for operators such as minus. H4 is a written operational-expression grammar, not a word morphology. Comparison is at component/position architecture level unless a sufficiently complete expression corpus can be transcribed from the manuscript.

## Frozen comparison levels

### Level A — architecture metrics (all systems where defined)

A1. number of typed slots/dimensions per encoded object;
A2. whether order/position is semantic;
A3. whether components are reusable across multiple objects;
A4. whether omission is licensed;
A5. whether recursion is licensed;
A6. whether interpretation is context-dependent;
A7. whether the mechanism is productive or a prepared lookup inventory;
A8. combinatorial capacity when it is defined by the source;
A9. fraction of theoretically possible slot combinations actually licensed/attested when computable;
A10. semantic typing of positions/components.

### Level B — abstract tuple metrics (only where actual tuple inventories can be recovered)

B1. slot marginal entropy;
B2. normalized conditional entropy `H(component | position) / H(component)`;
B3. pairwise slot mutual information;
B4. total correlation / multi-information;
B5. component reuse rate;
B6. one-component-neighbour graph: mean degree, median degree, isolated fraction, largest-connected-component fraction;
B7. family-size distribution under shared `n-1` components;
B8. observed tuple inventory / Cartesian capacity.

### Level C — surface string metrics (only for historically specified strings)

C1. character length distribution;
C2. prefix/suffix entropy by position;
C3. positional character conditional entropy;
C4. exact Levenshtein-distance-1 graph topology;
C5. repeated-substring/component recurrence;
C6. type-frequency and burstiness where token frequencies are historically defined;
C7. Markov/bigram predictability;
C8. component-boundary recoverability from surface strings.

A metric is `UNAVAILABLE` rather than imputed if the source does not supply the required data.

## Voynich target extraction

Only after this file and `historical_grammars_v1.json` are committed may new target aggregates be queried.

Primary target is RF exact-letter tokens from the already audited `voynich_semantic_notation_v1` Supabase schema. STA-family/full-STA/AAA are robustness representations, not tuning views.

No new hand-selected Voynich segmentation is allowed. The existing outcome-blind exhaustive edit-1 and affix/component infrastructure is reused for family/core and positional morphology statistics.

Frozen Voynich target panel:

V1. token count, type count and length distribution;
V2. edit-1 pair count and edit-position distribution;
V3. degree / connected-component topology of the edit-1 type graph;
V4. prefix/suffix candidate support and family-size distribution;
V5. character positional entropy/conditional entropy;
V6. prefix/suffix entropy asymmetry;
V7. component reuse across residual cores;
V8. observed component/core combinations relative to locally available alternatives;
V9. first-order character Markov predictability as a surface baseline;
V10. section/quire stability of the above where existing results permit it.

## Controls

Historical mechanisms are compared against:

- C0: simple iid character/token baseline matched only on output count and mean length;
- C1: independent fixed-slot tuples with the same number of slots and slot inventory sizes as the historical system, when inventory sizes are known;
- C2: slot-marginal shuffle of the historical tuple inventory, when the actual inventory is available;
- C3: prepared lookup-list control with the same number of outputs but unique components, to distinguish compositional reuse from mere codebook structure.

No control is tuned to maximise similarity to Voynich.

## Decision rule

There is no single omnibus `match/no-match` threshold because some historical systems are spatial and some linear. Conclusions are graded:

- **STRUCTURAL TRANSFER**: a source-faithful historical system matches multiple preregistered Voynich invariants and beats the relevant controls without parameter fitting.
- **PARTIAL TRANSFER**: at least one nontrivial invariant transfers, but major preregistered properties fail or are unavailable.
- **MECHANISM ONLY**: historical plausibility is established but the attested mechanism does not generate/test the Voynich surface properties without additional assumptions.
- **MISMATCH**: the historical mechanism predicts structural properties clearly unlike Voynich on the metrics it genuinely licenses.

A result cannot be promoted above `MECHANISM ONLY` by inventing a missing serialization or by combining separate historical systems post hoc.

## Immediate execution order

1. Freeze machine-readable source grammars.
2. Compute source-only architecture descriptors.
3. Recover exact tuple/codeword inventories where accessible.
4. Query the frozen Voynich target panel.
5. Run available Level-B/C comparisons and controls.
6. Record unavailable comparisons explicitly.
7. Close with a revised historical-mechanistic verdict.