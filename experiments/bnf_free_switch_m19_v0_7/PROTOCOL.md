# BnF 7342 free-switch unmarked-number model M19 v0.7 — preregistration

Date: 2026-08-09
Status at freeze: no v0.7 control or Voynich score observed.

## Hypothesis

At each plaintext letter the encoder may choose any of that letter's five BnF table values, but the table identity is not transmitted. Only the **unmarked numerical value** is represented on the page. Thus the receiver faces genuine contextual ambiguity.

This removes the deterministic-table schedule rejected in v0.6 while retaining the exact numerical repertoire of the five BnF tables.

## Exact numerical channel

The 19 unmarked values are:

`0,1,2,3,4,5,6,7,8,9,10,12,16,20,22,23,24,28,30`.

For plaintext letter `l`, its admissible value set is the union of the values assigned to it by F/M/G/L/H. Examples of exact anchors:

- `a → {1}`
- `b → {2}`
- `d → {4}`
- `o → {6,22}`
- `f → {6,8,28}`
- `n → {1,5,12,23}`
- `s → {1,6,30}`
- `y → {0,3,8}`

Conversely, values 0,22,23,28,30 are singleton plaintext anchors for `y,o,n,f,s` respectively; 9,16,20,24 have two compatible letters; low values are more ambiguous.

## Surface homophones

Primary ZLZI has 25 lowercased literal glyph labels. M19 therefore assigns:

- every one of the 19 numerical values to at least one surface glyph;
- exactly six numerical values have a second surface homophone;
- no value has more than two surface glyphs;
- every surface glyph has exactly one global numerical value.

Thus the fitted mapping is a surjection from 25 surface glyphs onto the exact 19-value BnF repertoire with multiplicities 1 or 2.

No null glyphs, changing mappings, section keys, transposition, syllabic units or table markers are permitted in v0.7. Word spaces are preserved.

## Encryption law for controls

For each plaintext letter, choose uniformly among that letter's **distinct** BnF values. Given a selected numerical value, choose uniformly among its available surface homophones.

Synthetic controls use all 19 values. One surface form is allocated to every value; the six numerically most frequent values in that control's training plaintext receive the second homophone. The 25 surface forms are then opaquely permuted. The generator requires all 25 surface forms to occur during training.

## Language model and induced numerical model

Frozen language panel: Latin, Italian, German, French, Ancient Greek, Hebrew, Arabic, Spanish/Castilian. Sources and normalization remain those used in v0.3–v0.6.

For each language estimate smoothed:
- plaintext letter unigram probabilities;
- word-initial and word-final probabilities;
- character bigram transition probabilities.

The encryption emission law is known from the frozen BnF tables:

`P(value | letter) = 1 / |V(letter)|` for `value ∈ V(letter)`, otherwise zero.

From the language model plus this emission law derive:
- numerical word-initial probabilities;
- numerical word-final probabilities;
- an induced 19×19 numerical bigram transition matrix, integrating over hidden plaintext letters.

The surface→number mapping is scored by observed surface-symbol word-initial, word-final and within-word bigram counts. This is a proper pairwise generative numerical-channel score, not the uniform candidate-set average that failed to qualify M57 v0.5.

## Mapping optimizer

Mapping multiplicities remain exactly 13 singleton values + 6 duplicated values = 25 surface glyphs.

Seeded simulated annealing uses only two legal moves:
1. swap the values assigned to two surface glyphs;
2. move the duplicate status from one value to another (a glyph assigned to a count-2 value changes to a count-1 value), preserving every value at least once and max multiplicity 2.

A deterministic coordinate/swap polish follows annealing. Mapping is fitted on training data only.

## Positive controls

Targets: Latin, Italian, German, Hebrew; two replicates each = 8 controls.

Each control uses 45,000 plaintext letters for fit and 39,000 further letters for held-out scoring from material disjoint from LM training. Hebrew may append the independently sourced Sefaria `Mishneh Torah, Torah Study` text as previously frozen; it does not enter LM training.

For each control:
- generate an exact M19 ciphertext under the frozen law;
- fit under all eight candidate language models;
- rank languages by held-out mapping-permutation z using 500 seeded permutations of the fitted surface→value assignment;
- report held-out **weighted mapping accuracy**: fraction of held-out cipher positions whose fitted numerical value equals the true numerical value;
- report mapping agreement between independent fits to the first and second halves of the training ciphertext, weighted by full-training symbol frequency.

Qualification requires all:
- Q1 correct language rank 1 in 8/8 controls;
- Q2 median held-out weighted mapping accuracy >=0.95;
- Q3 minimum mapping accuracy >=0.85;
- Q4 median target-language held-out permutation z >=10;
- Q5 median half-fit mapping agreement >=0.90 and minimum >=0.75.

If any criterion fails, no Voynich inference is permitted.

## Voynich split

Primary transcription: ZLZI.

- deterministic whole-folio 80/20 split by SHA-256 `20260809|M19|folio`;
- deterministic whole-folio training sample until >=45,000 non-space glyph positions;
- all held-out folios used for evaluation;
- lowercased alphabetic glyph labels only;
- mapping coverage must be >=99%.

For each language fit the M19 mapping on training only and compute held-out permutation z with 1,000 mapping permutations.

## Primary Voynich criterion

A language becomes a candidate only if:
- held-out mapping z >=10;
- margin over second-ranked language >=5;
- independent half-training mapping agreement >=0.80;
- exact 19-value surjection / max-two multiplicity respected;
- held-out mapped-glyph coverage >=99%.

A candidate then triggers contextual plaintext decoding by word-level Viterbi under the language bigram model and known BnF emission law. It must additionally achieve lexical enrichment z >=5 against 128 mapping-permutation decodes.

Finally transfer the literal ZLZI glyph→number mapping unchanged to TTLI and VDRB. Confirmation requires candidate rank 1, z>=7, lexical z>=3 and >=90% shared-glyph coverage in both.

Only all-stage success yields `CONFIRMED M19 SIGNAL`.

## Scope

A positive result would show compatibility with the free-switch unmarked-number mechanism, not prove that BnF lat. 7342 was used historically. A qualified negative rejects only this exact global M19 model.
