# VSN-B2 Transparency & Reproducibility Ledger

Date: 2026-08-12
Status: retrospective transparency repair + forward protocol

## Why this file exists

The early VSN-B2 chat reports exposed results and commit hashes but not enough of the executable machinery. This file makes the evidential chain auditable.

From this point forward, every material VSN-B2 result must expose:

1. exact or canonical executable code/SQL;
2. source URLs and immutable hashes where available;
3. PRNG seeds / seed namespace;
4. sample construction and exclusions;
5. null/control definitions;
6. compute job IDs and termination state;
7. failed/aborted analyses;
8. any correction that supersedes an earlier result;
9. a separation between historical-source claims, implementation choices, statistical results, and interpretation.

## Repository location

Branch: `experiment/voynich-semantic-notation-v1-20260812`

Primary experiment directory:
`experiments/voynich_semantic_notation_v1/`

Reproduction directory:
`experiments/voynich_semantic_notation_v1/repro/`

## Files in this reproduction package

- `run_matteo_surface_v1.py` — exact Python body recovered from HF job `6a7bbf9527caad61c6eaca79`.
- `run_matteo_k2_20seed_v1.py` — exact Python body recovered from HF job `6a7bbfc8f6d0f3ee953aa36a`.
- `run_source_latin_control_v1.py` — exact Python body recovered from HF job `6a7bbfedf6d0f3ee953aa372`.
- `run_matteo_section_nulls_v1.py` — exact Python body recovered from HF job `6a7bc266f6d0f3ee953aa37b`.
- `section_line_analysis_v1.sql` — canonical executable SQL reconstructed from PostgreSQL `pg_stat_statements` records. The SQL structure is recovered from PostgreSQL; literal constants were normalized by PostgreSQL to `$1`, `$2`, etc., so restored literal values are explicitly marked as reconstructed.
- `SOURCE_LEDGER.md` — historical and computational sources with provenance status.

## Exactness classes

### Class E — exact recovered executable

The four Python scripts above are copied from Hugging Face job metadata. They are the exact Python program bodies executed by the jobs, reformatted from `python -c` strings into `.py` files without algorithmic alteration.

### Class N — normalized recovered SQL

PostgreSQL retained the SQL structures in `pg_stat_statements`, but PostgreSQL normalizes constants to bind-like placeholders (`$1`, `$2`, ...). Therefore the recovered SQL structure is exact but literal constants are not recoverable byte-for-byte from `pg_stat_statements` alone.

`section_line_analysis_v1.sql` restores the constants from the query semantics and returned results. It is therefore a **canonical executable reconstruction**, not falsely presented as a byte-identical transcript of the original connector payload.

## Historical-source layer

### Matteo da Verona

Mechanisms tested:
- construct an artificial word from first syllables of selected source words;
- varying compound sizes are attested in the source tradition;
- separate medical/state encoding uses typed classes and degrees.

Strong scholarly anchor used in the deep research:
Valentina Cacopardo, *Memory and Imagination in the Ars Memorativa in Fifteenth-Century Italy* (PhD thesis, School of Advanced Study, University of London, 2021):
https://sas-space.sas.ac.uk/9846/

This institutional-repository thesis independently establishes the early-fifteenth-century Padua/Mantua/Venice ars-memorativa cluster and treats Matteo da Verona, Bartolomeo da Mantova and Jacopo Ragona.

A modern critical-edition transcription of Matteo was also consulted through an online copy of *Documenta Mnemonica*, Band III. The accessible copy used during discovery was not an ideal publisher-hosted source. **Before publication-quality quotation or philological claims, the passage must be rechecked against a library/publisher copy or manuscript witness.** The statistical experiment depends on the abstract rule, not on an unverified modern spelling of any Voynich token.

### Bartolomeo da Mantova / Jacopo Ragona

Main scholarly source:
Cacopardo 2021, institutional repository above.

They are treated as mechanism precedents only. No free Cartesian-product grammar was inferred from them.

### Vat. lat. 10488

PAL/Jordanus manuscript record:
https://ptolemaeus.badw.de/jordanus/ms/10273

The record gives Venice, 1424, Italian, mathematics. It is used as a historical operational-notation comparator, not as a source for the K2 generator.

## Computational-source layer

### Latin vocabulary

PyWORDS repository:
https://github.com/sjgallagher2/PyWORDS

Exact file used:
`pywords/data/lingualatina_voclist.txt`

Git blob recorded at preregistration:
`5dc8e924f253ef18cc72d72daa15ec49a805b8f8`

Raw-byte SHA-256 observed in the primary run:
`5a139a6e7a3b9bfe9ef0b0e98e5178fb1c42be66dc3034c3f6f5e3d91b099b9c`

Source lines: 1,902.
Eligible normalized unique words in the run: 1,846.
Derived unique first syllables: 429.

PyWORDS describes itself as a Python toolkit based on Whitaker's WORDS. This vocabulary is a convenience lexical source for instantiating the historical operation; it is **not** claimed to represent the vocabulary actually used by Matteo or a Voynich author.

### Voynich RF representation

Frozen Reference Transliteration source used elsewhere in VSN/STA work:
https://voynich.nu/data/sta/RF1b.txt

SHA-256:
`81c331b7d8e76761e27d350c3b37ccfbe192848e6c8a227bcb5d40fb29259b17`

Primary RF token inventory for VSN-B2 comes from Supabase table:
`voynich_semantic_notation_v1.rf_token_types`

Occurrence table:
`voynich_semantic_notation_v1.rf_occurrences`

Original edit-path table:
`voynich_semantic_notation_v1.rf_edit1_pairs`

Important correction:
- edit-path rows = 28,435;
- distinct unordered one-edit token pairs = 27,307;
- duplicate/alternative paths = 1,128.

All section/line hierarchy results use distinct unordered pairs.

## K2 implementation choices

These are **our implementation choices**, not claims copied verbatim from Matteo:

- `j -> i` normalization;
- vowel nuclei `a e i o u y`;
- diphthongs `ae au oe ei eu ui` treated as one nucleus;
- an orthographic first-syllable splitter using intervocalic consonant and mute+liquid rules;
- uniform sampling over normalized source lemmas for the primary generator;
- duplicate generated surfaces discarded until the requested unique-type count is reached.

Pre-target sanity examples:
`tripode -> tri`, `pepo -> pe`, `corvus -> cor`, `vetula -> ve`.

These rules were frozen before the main target comparison in `MATTEO_SIMULATION_SPEC_V1.md`.

## Primary generator regimes

All were reported; K2 was not the only run:

- K2: exactly 2 first syllables;
- K3: exactly 3;
- K5: exactly 5;
- KMIX: equal-probability choice among 2, 3, 5.

Robustness also sampled uniformly over unique first syllables rather than source lemmas.

## Controls

Primary hostile controls:

1. iid uniform characters over the Latin-source alphabet;
2. iid characters drawn from the Latin-source character marginal;
3. raw source-Latin vocabulary edit topology;
4. 20 independent K2 seeds;
5. matched-size K2 simulations per Voynich section;
6. section-conditioned line shuffles preserving the actual Voynich token multiset and every running-text line length.

## Compute ledger

Completed CPU-only Hugging Face jobs:

- `6a7bbf9527caad61c6eaca79` — primary K2/K3/K5/KMIX + iid controls; Python 3.12; timeout 600 s; runtime ~2 s.
- `6a7bbfc8f6d0f3ee953aa36a` — 20-seed K2 robustness; CPU; timeout 600 s; runtime ~4 s.
- `6a7bbfedf6d0f3ee953aa372` — raw Latin vocabulary control; CPU; timeout 300 s; runtime <1 s.
- `6a7bc266f6d0f3ee953aa37b` — matched-size section K2 nulls; CPU; timeout 600 s; runtime ~3 s.

A separate STA/AAA robustness job is documented in `HISTORICAL_GRAMMAR_REPRESENTATION_ROBUSTNESS_V1.md`.

Final HF status after these runs: no running jobs.

## Failed / corrected analyses

These are part of the evidential record and must not be erased:

1. A preliminary partial-vocabulary K2 simulation suggested the wrong prefix/suffix direction. The full 1,846-word run overturned that provisional result. It is superseded.
2. Early reporting treated 28,435 edit-path rows as distinct pairs. Audit found 27,307 distinct pairs. Section/line analysis uses the corrected definition.
3. An early line query used an `OR` join to the edit-path table and could duplicate opportunities/hits where multiple edit paths represented one pair. It was rejected and replaced by a deduplicated unordered-pair CTE.
4. Cosmological line clustering was initially extreme. Layout decomposition showed it is dominated by circular/diagram loci, especially f57v `<f57v.3,+Cc>`. Running-text Cosmological does not survive the hostile line shuffle.
5. A 256-permutation combined Supabase query hit statement timeout. It was read-only, changed no data, and is not reported as a completed test. The completed 64-permutation test is the result of record.

## Current interpretation boundary

Supported:
- literal K2 first-syllable composition creates a global edit-location topology unexpectedly close to aggregate Voynich and to some sections;
- it fails section-local edit density and stronger Voynich positional/transition constraints;
- Voynich one-edit neighbours are locally concentrated in ordinary running-text lines in multiple sections even after within-section token shuffling.

Not supported:
- Matteo K2 is the Voynich generative mechanism;
- any Voynich glyph/token has been assigned a Matteo meaning;
- the Voynich manuscript is a mnemonic treatise;
- Padua origin is proven;
- Ragona/Bartolomeo may be freely combined with Matteo to repair failures.

## Forward transparency rule

Before the proposed state-gated K2 experiment is run, its full code, parameter grid, seed namespace, section holdout policy, nulls, and stopping/acceptance rules will be committed **before target scoring**. Any post-freeze code change will require a new version and explicit reason.