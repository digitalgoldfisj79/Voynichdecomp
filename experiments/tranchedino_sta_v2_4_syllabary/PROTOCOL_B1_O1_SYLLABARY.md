# Tranchedino × STA v2.4 — B1-O1 syllabary component-oracle qualification

Date frozen: 2026-08-09
Namespace: `TRANCHSTA24B1O1`
Parent: `STAGE B0 PASS`
Status: **PROSPECTIVE COMPONENT QUALIFICATION — NO VOYNICH FIT AUTHORISED**

## 1. Question

Given the exact one-sign f.134v–135r historical mechanism, is the new 64-entry syllabary itself identifiable from finite Paduan ciphertext when every non-syllabic part of the key is supplied as an oracle?

This gate isolates the novel component before any attempt to infer alphabetic, null, geminate, lexical and syllabic assignments jointly.

Failure closes v2.4 before a joint solver is built.

## 2. Prospective qualification source

The first 1,248 held-out Paduan line records were consumed during development and are permanently D1-contaminated.

The untouched Q1-O1 tail is frozen as the remaining:

- 175 line records;
- 6,619 normalised 19-letter characters;
- pages 243–251.

No recovery metric or decoded text from this tail was generated before this protocol freeze.

The frozen language-training source remains the old pre-page-183 partition. No modern corpus is admitted.

## 3. Qualification controls

Generate twelve fresh controls over the complete untouched Q1-O1 source, one for every cell of:

- `p_syll ∈ {0.25,0.50,0.75,1.00}`;
- `p_null ∈ {0.00,0.03,0.10}`.

Each cell receives an independent:

- 166-label surface permutation;
- syllable-use Bernoulli stream;
- alphabetic-homophone choices;
- null insertion stream.

The generator is otherwise exactly the frozen B0 generator.

## 4. Oracle information

The solver receives:

- the exact surface-symbol class partition;
- the true semantic mapping for all observed alphabetic, geminate, null and secure lexical signs;
- the complete fixed list of 64 historical syllable plaintext units;
- the cell's `p_syll` nuisance stratum.

It does **not** receive the mapping from any syllable surface sign to a syllable plaintext unit.

Unobserved historical syllable signs remain in the 64-way permutation but do not count as recovery failures.

This is intentionally generous. The purpose is component identifiability, not a final blind decipherer.

## 5. Frozen proposal model

Construct a semantic-unit vocabulary consisting of:

- 19 literal letters;
- 8 historical geminates;
- 64 historical syllables;
- 9 securely transcribed lexical entries.

For each `p_syll` cell, generate four deterministic stochastic tokenisations of every training line using the frozen historical encoding order but without cipher labels or null tokens. Accumulate additive-0.25 dense semantic-unit trigram counts and convert to conditional log probabilities with line reset.

Nulls are removed from the oracle semantic sequence because their true identity is supplied.

Frequency initialisation ranks observed opaque syllable signs by occurrence count and assigns them to historical syllable units ranked by two independently seeded training-tokenisation passes. This initialisation is a search proposal only; qualification is based on the recovered true mapping, never on frequency score.

## 6. Frozen search

For each control run two independent ensembles A and B.

Each ensemble:

1. starts from the frozen frequency initialisation;
2. performs exactly 50,000 random pair-swap proposals over the complete 64-way syllable permutation;
3. uses simulated annealing on the semantic-unit trigram score;
4. temperature decreases geometrically from 3.0 to 0.01 over the 50,000 proposals;
5. uses an independent deterministic SHA-256-derived random seed;
6. retains the highest-scoring mapping encountered.

Only trigram terms whose local token neighbourhood changes under a proposed swap are rescored. This is an implementation optimisation and must reproduce full rescoring exactly.

No manual mapping inspection or plaintext-string inspection is permitted on Q1-O1.

## 7. Metrics

For each ensemble/control emit only numerical truth-based metrics:

- number of observed syllable identities;
- syllable occurrence count;
- occurrence-weighted syllable semantic recovery;
- observed-identity syllable mapping recovery;
- expanded plaintext edit accuracy after applying the oracle non-syllable map and recovered syllable map;
- final trigram score per non-null cipher event.

For each control also emit A/B:

- occurrence-weighted semantic-map agreement over observed syllable signs;
- absolute final-score difference per non-null event.

No decoded Q1-O1 string may be printed or archived.

## 8. Binding qualification gates

Every one of the 12 controls must have at least 45 observed syllable identities and at least 400 syllable occurrences. Otherwise the qualification source is declared underpowered and the result is `B1-O1 SOURCE INSUFFICIENT`, not a cipher-family failure.

If the source is sufficiently occupied, qualification requires all of:

- median occurrence-weighted true syllable recovery `>=0.95`;
- minimum occurrence-weighted true syllable recovery `>=0.85`;
- median observed-identity mapping recovery `>=0.90`;
- minimum observed-identity mapping recovery `>=0.75`;
- median expanded-plaintext edit accuracy `>=0.97`;
- minimum expanded-plaintext edit accuracy `>=0.93`;
- median A/B occurrence-weighted map agreement `>=0.95`;
- minimum A/B occurrence-weighted map agreement `>=0.85`.

The score-difference statistic is diagnostic because equal/near-equal likelihoods can arise from rare unobserved assignments; no post-hoc score threshold is added.

Any failed recovery/agreement gate gives:
`B1-O1 SYLLABARY COMPONENT NOT QUALIFIED`.

A full pass gives:
`B1-O1 SYLLABARY COMPONENT QUALIFIED`.

## 9. Advancement

A B1-O1 pass authorises design of the next component gate, which must remove the true class partition and/or other oracle mappings prospectively.

It does **not** qualify a full f.134v–135r solver, does not establish source specificity, and does not authorise any Voynich target scoring.

Before any final blind qualification can authorise a target, the programme must additionally obtain a fresh plaintext qualification source not used during solver development, or explicitly close with `FRESH-SOURCE QUALIFICATION UNAVAILABLE`.
