# BnF 7342 marked single-value ambiguous-code programme v0.5 — preregistration

Date: 2026-08-09
Status at freeze: no v0.5 Voynich language score observed.

## Motivation

Earlier programmes adapted the five BnF lat. 7342 letter-value tables into **injective pairs**. That construction is mathematically valid but not what the manuscript literally presents. The manuscript instead gives five separate many-to-one letter→number alphabets.

v0.5 tests a narrower adaptation of the literal apparatus:

**Voynich surface glyph → one marked `(table, numerical value)` code → one of the plaintext letters sharing that value in that table.**

Thus the reverse map is intrinsically ambiguous and language context must resolve it. This is not equivalent to the v0.3/v0.4 direct glyph→letter substitution tests.

## Exact BnF code inventory

Plaintext alphabet:

`abcdefghiklmnopqrstuxyz`

The five frozen tables are F, M, G, L, H with the values transcribed in the parent programme. Within each table, identical numerical values collapse to one marked code. This produces exactly **57 marked codes**:

- F: 15 distinct values
- M: 15
- G: 8
- L: 10
- H: 9

A marked code retains table identity: `G:1` is distinct from `H:1`.

Across the 57 codes, candidate plaintext-set sizes are prospectively known: 23 singleton codes, 19 size-2, 11 size-3, one size-4, one size-5, two size-6.

No code values or candidate sets may change after scoring.

## Cipher model M57

- Each lowercased Voynich transliteration glyph label is assigned to exactly one of the 57 marked codes.
- The assignment is **global and injective** over surface glyph labels: two distinct Voynich glyph labels may not represent the same marked code.
- A cipher occurrence may decode contextually to any plaintext letter in its marked code's candidate set.
- Word spaces are preserved and treated as observed boundaries.
- No nulls, transposition, changing keys, section-specific keys, positional schedules, or unmarked-value collapse are allowed in v0.5.
- Train-unseen held-out glyph labels are hard breaks; mapped held-out glyph coverage must be >=99%.

This is deliberately more constrained than allowing arbitrary per-occurrence table selection.

## Language panel

Frozen eight-language panel:

Latin, Italian, German, French, Ancient Greek, Hebrew, Arabic, Spanish/Castilian.

Normalization remains the parent programme's 23-letter Latin alphabet with `j→i`, `v/w→u`, using frozen romanization through `Unidecode` for non-Latin scripts.

Language-model sources remain the same UD corpora used in v0.3/v0.4. Positive-control target languages are Latin, Italian, German and Hebrew. Hebrew known-plaintext controls may append the independent Sefaria `Mishneh Torah, Torah Study`, Hebrew, `Torat Emet 363` source after the disjoint UD Hebrew holdout, as already prospectively documented in v0.4 Amendment 002. The Sefaria text does not enter LM training.

## Compatibility scorer

The primary solver uses a smoothed **word-internal character-bigram compatibility score**, not an unconstrained plaintext generator.

For each language:

1. Estimate smoothed `P(next_letter | previous_letter)`, `P(word_initial_letter)` and `P(word_final_letter)` on the frozen training corpus.
2. For a marked code `c`, define `C(c)` as its BnF-allowed plaintext-letter set.
3. The compatibility probability for adjacent codes is the arithmetic mean of the language transition probabilities over all allowed plaintext pairs in `C(c1) × C(c2)`.
4. Initial/final compatibility is the arithmetic mean over the allowed set.
5. Score a ciphertext mapping by the observed cipher-symbol bigram, word-initial and word-final counts. All scoring is normalized per scored event.

The mapping is fitted on training ciphertext only using seeded simulated annealing followed by deterministic coordinate polish. The held-out mapping is never refitted.

## Positive-control construction

For each target language, two independent controls are generated (8 total).

Each control:

- uses 45,000 plaintext letters for mapping fit and 39,000 further plaintext letters for held-out scoring;
- draws from language material disjoint from LM training;
- selects exactly 25 distinct marked BnF codes by deterministic rejection sampling until all 23 plaintext letters have at least one selected compatible code;
- encrypts each plaintext letter by uniformly choosing among selected marked codes that contain that letter;
- applies a seeded opaque permutation from the selected 25 codes to surface-symbol IDs 0–24;
- requires all 25 surface IDs to occur in the training segment; otherwise the deterministic generator advances to the next attempt.

The true marked-code assignment is retained only for control diagnostics.

## Positive-control qualification

For every control, fit M57 separately under all eight candidate language models and rank languages using held-out mapping-permutation z.

The held-out null uses 1,000 seeded permutations of the fitted 25-code assignment across the 25 surface glyph labels while preserving the selected code multiset and all ciphertext order.

A control also reports **true-letter compatibility accuracy**: the fraction of held-out plaintext positions whose true letter belongs to the fitted marked code's allowed set.

The instrument is qualified only if all are true:

- Q1: correct language ranks first in **8/8** controls;
- Q2: median true-letter compatibility accuracy >=0.95;
- Q3: minimum compatibility accuracy >=0.85;
- Q4: median target-language held-out permutation z >=10;
- Q5: every control has mapped held-out symbol coverage >=99%.

If this gate fails, no Voynich language inference is allowed.

## Voynich split

Primary transcription: ZLZI from `main/voynich_transcriptions_slim.json`.

- Whole folios are assigned deterministically 80/20 by SHA-256 of `20260809|M57|folio`.
- From training folios, choose whole folios in deterministic hash order until at least 45,000 non-space glyph positions are accumulated.
- Held-out evaluation uses all mapped positions on the held-out folios.
- Lowercase the single ZLZI capital-I occurrence.
- Mapping is fitted only on the representative training folios and frozen before held-out scoring.

## Voynich primary signal

For each language, report held-out mapping-permutation z using 1,000 mapping permutations, plus held-out normalized compatibility score.

A language becomes an M57 candidate only if:

- z >=10.0;
- z-margin over the second-ranked language >=5.0;
- fitted mapping is injective and uses only the 57 exact BnF marked codes;
- >=99% of held-out glyph positions are mapped.

A candidate then triggers a second-stage contextual decoder, still without refitting the code assignment:

- exact/beam Viterbi under the language's character model over each marked code's allowed plaintext set;
- lexical enrichment against mapping-permutation decodes;
- lexical z must be >=5;
- TTLI and VDRB transfer using the literal ZLZI glyph→marked-code assignment unchanged; candidate language must remain rank 1 and exceed z>=7 on each with >=90% shared-glyph coverage.

Only if all candidate-stage conditions pass may the result be called a `CONFIRMED M57 SIGNAL`.

## Interpretation

A positive result would show compatibility with a marked single-value BnF-style ambiguous numerical code. It would not establish that BnF lat. 7342 itself was used or that the apparatus was historically intended for cryptography.

A qualified negative result rejects only this global marked-code model. It does not reject unmarked numerical values, table schedules inferred from position, changing codebooks, nulls, syllabic units, transposition, or non-linguistic generation.
