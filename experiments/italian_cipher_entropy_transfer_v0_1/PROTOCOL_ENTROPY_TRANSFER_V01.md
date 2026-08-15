# Northern Italian Cipher Entropy Transfer v0.1 — external-only

Date: 2026-08-15
Status: prospective external calibration. **No Voynich data may be loaded or scored in this experiment.**

## Question
Before asking whether a 1435–1448 Northern Italian diplomatic-cipher mechanism resembles the Voynich Manuscript, determine what each historically attested ingredient does to entropy on ordinary source texts.

The experiment estimates a transfer function

`plaintext -> cipher mechanism -> entropy change`

without target fitting. A later target comparison is permitted only after this transfer function is frozen and the atomic Voynich representation stack has separately passed integrity tests.

## Historical anchors
Primary in-window ingredients:

- dense homophonic substitution: Modena 1435 is reported with three or four substitution options for nearly the full 21-letter Italian alphabet;
- mixed multi-glyph / bigram-syllable units: Milan 1448 is reported with multiple glyph options plus two-part glyph units for plaintext bigrams/syllables;
- nomenclator/codeword elements and nulls are attested in the broader early Italian diplomatic tradition.

Exact null rates, nomenclator sizes, and proportions of two-glyph units are **not historically calibrated by the cited evidence**. Those quantities are therefore sensitivity parameters, never fitted to the target.

Alberti-style orthographic normalisation is retained only as a post-1460 sensitivity control and is not part of the primary pre-1450 mechanism.

## Source-text families
External texts only, inherited from the hostile v0.2 real-control bundle:

- OLD_ITALIAN
- LATIN_LLCT
- LATIN_PROIEL
- GERMAN_GSD
- BVGS_RECIPE (c. 1350 German recipe text)
- CREMMA_MEDICAL (medieval Latin medical material)

All are transformed in fixed plaintext-character windows before entropy estimation. No class-C/magic text is used to fit the cipher mechanism.

## Two representation levels
Every generated ciphertext is measured twice:

1. **ATOMIC** — each cipher unit is one symbol, even if written using two glyphs.
2. **GLYPH** — the actual written glyph sequence, flattening two-glyph units.

This is mandatory. Mixed-length codes can raise marginal glyph entropy while creating strong within-codeword dependencies that lower conditional entropy. Treating atomic units and written glyphs as the same representation would confound mechanism with tokenisation.

## Entropy metrics
For each plaintext window and generated ciphertext:

- alphabet size `K`
- marginal entropy `H0 = H(X_t)`
- first-order conditional entropy `H1 = H(X_t | X_{t-1})`
- second-order conditional entropy `H2 = H(X_t | X_{t-2}, X_{t-1})`
- lag-1 mutual information `MI1 = H0 - H1`
- normalised conditional entropy `H1/log2(K)`
- output symbols per plaintext character
- `H1 * output_length / plaintext_characters` as an information-production-rate diagnostic

Primary interpretation uses paired deltas against the same plaintext window under identity coding. H2 is diagnostic because sparse high-order contexts can be biased.

## Mechanism battery
One factor at a time plus fixed combined variants:

- ID: identity / one-symbol substitution control
- HOM2: two homophones per plaintext letter
- HOM34: three or four homophones per plaintext letter (historically anchored 1435-style density)
- HOM34+NULL: null insertion at 1%, 2.5%, 5% sensitivity rates
- NOM: top 25, 50, 100 word types replaced by atomic codewords
- BIGRAM: top 10, 20, 40 training-window bigrams replaced by atomic code units
- BIGRAM+DIGLYPH: BIGRAM20 with 25%, 50%, 75% of code units written as two glyphs
- COMBINED_LOW/MID/HIGH: fixed combinations spanning the above sensitivity ranges
- ORTHO_POST1460: double-letter collapse, `qu -> q`, `h` omission; secondary only

No parameter is tuned after looking at target data.

## Randomisation
Homophone choices and null insertion are stochastic. Each plaintext window is transformed under six deterministic seeds. Results are paired within family/window/seed.

## Sanity gates
The external run is invalid if any of the following fail:

1. ID atomic and ID glyph metrics differ beyond floating tolerance.
2. A bijective one-symbol recoding changes entropy beyond floating tolerance.
3. HOM34 lowers median H0 in more than one source family.
4. BIGRAM20+DIGLYPH50 fails to produce a measurable atomic-vs-glyph conditional-entropy difference in at least four source families.
5. Fewer than four source families provide at least four qualifying plaintext windows.

## Stability report
For each mechanism and representation, report family-level median entropy deltas and the cross-family median. A directional claim is called `STABLE` only when at least 4/6 usable source families have the same sign and the cross-family median magnitude exceeds 0.05 bits/symbol.

No “best mechanism” is selected in this experiment. The output is a transfer map, not a target-ranking exercise.

## Scientific firewall
- Voynich is not downloaded.
- No Voynich entropy value appears in the protocol, code, thresholds, parameter grid, or decision logic.
- Naibbe is not used for fitting or calibration because it is target-informed.
- Historical presence and historical parameter calibration are distinguished explicitly.
- A later target experiment must use the repaired atomic representation stack and cannot change this mechanism grid after target exposure.
