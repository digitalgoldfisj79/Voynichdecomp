# BnF 7342 numerical-alphabet global-key programme v0.3

Date frozen: 2026-08-08
Base: positive-control v0.2 result commit `1bdd89f518c88d3a165febc48c6e20523c0f3072`
Status at freeze: **VOYNICH GLOBAL RESULT UNSEEN**

## Question

Could a single fixed homophonic key, of the class obtainable by rendering BnF lat. 7342 table-pair codes as opaque glyphs, explain the Voynich glyph stream across the manuscript?

This is deliberately narrower than arbitrary table switching. A cipher glyph has one plaintext-letter value globally. Spaces are treated as preserved plaintext word boundaries. Failure therefore rejects only this fixed-key, one-cipher-glyph-per-plaintext-letter realization, not every conceivable use of the BnF tables.

## Pre-result corpus census

The census is an input audit, not a language result. From `main/voynich_transcriptions_slim.json`:

- ZLZI: 226 pages; 37,465 word tokens; 7,598 word types; 186,727 non-space glyph positions; 26 literal characters, 25 after lowercasing the single capital `I`.
- TTLI: 225 pages; 34,351 tokens; 170,585 glyph positions; 21 literal characters.
- PCCA: 131 pages; 16,110 tokens; 66,175 glyph positions.
- FFSG: 202 pages; 33,095 tokens; 141,850 glyph positions.
- GCGA: 225 pages; 39,535 tokens; 153,773 glyph positions.
- VDRB: 226 pages; 34,038 tokens; 170,017 glyph positions.
- RGVN: 153 pages; 20,059 tokens; 84,872 glyph positions.
- 131 pages contain all seven fixed transcription families.

A global word-type-as-one-homophone model is not admissible under the exact BnF codebook sizes: ZLZI alone has 7,598 word types versus 92 T2 / 199 T3 possible marked BnF codes. The global test therefore concerns **glyph-level substitution**.

## Plaintext alphabet and languages

The frozen 23-letter alphabet is `abcdefghiklmnopqrstuxyz`, with `j -> i`, `v -> u`, `w -> u`. Spaces are retained.

Candidate language panel, unchanged in spirit from v0.1/v0.2:

1. Latin
2. Italian
3. German
4. French
5. Greek
6. Hebrew
7. Arabic
8. Castilian/Spanish

Greek, Hebrew and Arabic are deterministically romanized before 23-letter normalization. These are language-family comparators, not claims that a historical encoder used those exact romanizations.

## Language models

Character 4-gram models with additive smoothing are trained from the same fixed public corpora used in v0.2: Latin ITTB, Italian ISDT, German GSD, French GSD, Ancient Greek Perseus, Hebrew HTB, Arabic PADT, Spanish AnCora. Spaces are included as a 24th symbol in v0.3.

For Latin, Italian, German and Hebrew, every fifth sentence is held out from LM training for positive controls. No positive-control plaintext may enter its target LM training set.

## BnF-derived capacity models

T2: the four globally injective table pairs FM, FG, FL, ML give four possible marked homophones per plaintext letter. Thus at most four observed cipher glyphs may map to one plaintext letter.

T3: all locally unique marked pair-values are allowed; per-letter capacities are the frozen v0.2 values (6-10, total codebook size 199).

Because Voynich ZLZI has only 25 lowercased cipher glyph types, T3 is principally a flexibility diagnostic; T2 is the primary historical-conservative model.

The actual numeric pair labels are opaque after rendering. Therefore the inverse problem depends on the per-letter homophone capacities, not on the visual spelling of numeric code labels.

## Solver

A capacity-constrained global homophonic-substitution solver is used. One mapping `cipher glyph -> plaintext letter` is shared by every occurrence.

The solver uses:

- deterministic seeds derived from SHA-256;
- frequency-aware random initializations;
- simulated annealing with incremental 4-gram rescoring of only n-grams affected by a proposed one-symbol reassignment or capacity-forced swap;
- multiple independent restarts;
- final greedy single-symbol coordinate polishing;
- no dictionary, crib, semantic keyword, folio illustration, section label or Voynich-adjacent linguistic hypothesis.

Optimization may use a deterministic representative training sample capped at 45,056 non-space characters plus their spaces, drawn evenly across training pages. The learned mapping is always evaluated without further adjustment on held-out pages/full held-out text.

## Positive-control gate P0 — binding before Voynich

Synthetic controls render known plaintext through a single fixed 25-symbol global homophonic key satisfying T2 capacities. Word spaces are preserved. The 25 opaque symbols are assigned deterministically, with at least one code for every plaintext letter present where possible and remaining codes allocated to common letters within capacity. Encryption chooses among a letter's codes deterministically-randomly; the true key is retained only for scoring.

Controls: Latin, Italian, German, Hebrew; lengths 11,264 and 45,056 plaintext letters; two independent spans/keys per language. T2 solver is binding; T3 is diagnostic.

At length 45,056, T2 must satisfy all:

- P0.1 target language ranks first in **8/8** controls;
- P0.2 median held-out plaintext character accuracy >= 0.90;
- P0.3 minimum held-out character accuracy across the eight controls >= 0.75;
- P0.4 median learned-key held-out 4-gram score exceeds 100 capacity-preserving random mapping permutations by z >= 10.

If P0 fails, **Voynich optimization is prohibited** and the programme closes `INSTRUMENT NOT QUALIFIED`.

The 11,264-character tier measures the recoverability frontier but is not itself binding.

## Voynich corpus and holdout

Primary transcription: ZLZI, lowercased, spaces retained. Folios are assigned deterministically to train/holdout by SHA-256 of folio ID: hash modulo 5 == 0 is holdout; all others train. No character from a holdout folio may be used to optimize the mapping.

For each language and T2/T3, optimize one global key on training folios. Evaluate the fixed key on held-out folios.

For each fitted key calculate:

- held-out 4-gram score;
- z score against 100 permutations of the learned mapping assignments across cipher symbols, preserving the multiset of plaintext assignments;
- decoded plaintext-letter frequencies;
- capacity usage;
- held-out word samples (reported only as diagnostics, not used in selection).

## Predeclared Voynich signal criterion

A candidate is `GLOBAL_KEY_SIGNAL` only if all are true:

1. P0 passes.
2. Under T2, one language has held-out mapping-permutation z >= 10.
3. Its z exceeds the second-ranked language by >= 5.
4. The same fixed ZLZI mapping, applied by literal glyph label without refitting, has positive mapping-permutation z >= 5 on both TTLI and VDRB over pages they share with ZLZI holdout. Rare labels absent from the ZLZI key are excluded from transfer scoring and coverage is reported.
5. No more than four cipher glyph types map to any plaintext letter.

T3 alone can produce at most `FLEXIBILITY_ONLY`, never the primary signal.

Anything else is `NONRESOLVING` or `GLOBAL_FIXED_KEY REJECTED` depending on positive-control power and held-out separation.

## Interpretation rules

- A P0-qualified, strongly negative ZLZI result is evidence against this specific global fixed homophonic substitution class for the tested languages.
- It is not evidence against variable keys by folio/section, polyalphabetic switching, nulls, transposition, syllabic codes, abbreviation systems, or a non-linguistic generator.
- A positive statistical signal is not a translation. It licenses a second frozen stage for linguistic/philological validation of the decoded text.
- No plaintext excerpt may be called meaningful unless the predeclared statistical gate passes.
