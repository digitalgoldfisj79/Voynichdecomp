# PIIV-FIXED-POSITION frozen protocol

Seed: `20260714`.

## Historical hypothesis

Trithemius, *Polygraphia* IV, uses invented coverwords whose plaintext value is recovered from a fixed internal character, conventionally the second character. Applied to Voynich, the exact hypothesis predicts that one fixed EVA glyph position—primarily the second glyph from the left—forms a globally consistent monoalphabetic substitution of a natural-language character stream.

This is a different hypothesis from the Book III row-table test. It is tested before any Book IV coverword table is fitted or transcribed.

## Data

- Primary transcription: `ZLZI` from `voynich_transcriptions_slim.json`.
- Sensitivities: `ZLZB`, `TTIA`.
- Text-only folios and tokens containing characters outside `[a-z]` are excluded.
- Davis hand labels are sealed and unused.
- Grouping uses whole approximate quire blocks; no folio is divided across folds.

## Candidate streams

For every eligible Voynich token, extract exactly one character at each frozen position:

- from left: `F1` through `F6`;
- from right: `L1` through `L3`.

Tokens shorter than the requested position are omitted from that stream. Physical lines remain separate and language-model state resets at every line. `F2` is the preregistered Book IV position; the others control for positional fishing.

## External language models

Character language models are built only from external corpora:

- Latin: combined *Secreta Secretorum* and *Picatrix*;
- Old Italian: *Rettorica*.

Normalisation uses the 24-letter Trithemian alphabet: lowercase ASCII, `j→i`, `v→u`, and removal of non-alphabetic characters. No Voynich sequence is used to estimate a language model.

## Monoalphabetic recovery

For each candidate position and each grouped fold:

1. learn an injective symbol-to-letter substitution on training quires by simulated annealing over unigram and bigram external-language likelihood;
2. choose Latin or Italian using training score only;
3. evaluate the frozen mapping and chosen language on held-out quires using smoothed trigram bits per character;
4. calculate gain over the corresponding held-out unigram model;
5. record mapping agreement across folds.

## Calibration

### Positive controls

External Latin and Italian plaintext is cut to the Voynich line template, encrypted by a random monoalphabetic substitution, and passed through the identical recovery pipeline. The detector must recover language-like held-out gain and at least 70% of the planted mapping, allowing global alphabet symmetries only where letters are absent.

### Negative controls

For each candidate stream, 100 nulls independently permute extracted characters within each approximate quire while preserving:

- symbol frequencies by quire;
- line lengths;
- position eligibility;
- folio and section composition.

The complete best-position search is repeated for every null, so the family-wise p-value accounts for all nine positions.

## Decision

The exact fixed-position payload hypothesis receives a **PASS** only if all conditions hold:

1. the positive-control calibration passes;
2. `F2` is the best or statistically tied-best position in the primary transcription;
3. the family-wise permutation p-value for the observed `F2` held-out gain is `≤0.05`;
4. median fold-to-fold mapping agreement for `F2` is `≥0.70`;
5. both alternate transcriptions retain `F2` as best or tied-best with nominal `p≤0.05`;
6. the held-out gain is no more than 0.50 bits/character below the weaker positive-control gain.

If calibration passes but any criterion fails, verdict: **FAIL_EXACT_FIXED_POSITION_PAYLOAD**.

If positive calibration fails, verdict: **UNRESOLVED_CALIBRATION_FAILED**.

## Scope

A failure rejects a globally fixed one-glyph-per-word payload position under a monoalphabetic substitution, including the literal Book IV second-character mechanism. It does not reject changing positions, polyalphabetic payload mappings, deletions/insertions, multi-glyph payloads, scribe-specific keys, sparse messages, or an independent post-encryption surface realiser. Those are different and more complex hypotheses.
