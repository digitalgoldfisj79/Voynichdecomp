# VSN-B2-v1 — Matteo Artificial-Word Simulation Specification

Frozen: 2026-08-12 (Europe/London)

## Purpose

Test the literal surface consequence of Matteo da Verona's attested rule: select source words, take their first syllables, and concatenate those syllables in source order to form an artificial word. This is an instantiation of the attested mechanism, not a claim about which source vocabulary Matteo or a Voynich author used.

## Independent lexical source

Public PyWORDS/Whitaker-derived Latin vocabulary:

- repository: `sjgallagher2/PyWORDS`
- file: `pywords/data/lingualatina_voclist.txt`
- Git blob SHA: `5dc8e924f253ef18cc72d72daa15ec49a805b8f8`
- indexed line count at freeze: 1,902

The full file is used. No word is selected because of resemblance to Voynich.

Lexicon normalization is frozen as:

1. lowercase;
2. retain entries matching `[a-z]+` after lowercasing;
3. length >= 2;
4. contains at least one vowel from `aeiouy`;
5. deduplicate exact forms;
6. sample source words uniformly over the resulting unique vocabulary.

A robustness run samples uniformly over unique derived first syllables instead of uniformly over source words.

## First-syllable rule

Orthographic Latin approximation, validated before target comparison on the source examples `tripode -> tri`, `pepo -> pe`, `corvus -> cor`, `vetula -> ve`.

- vowel nuclei: `a e i o u y`;
- diphthongs treated as one nucleus: `ae au oe ei eu ui`;
- with a single intervocalic consonant, that consonant begins the next syllable;
- with a multi-consonant cluster, split before the final consonant, except a final mute+liquid cluster (`b/c/d/g/p/t/f/k + l/r`) remains the onset of the next syllable;
- `j` is normalized to `i` if encountered.

No syllable boundary is changed after looking at Voynich metrics.

## Slot-count regimes

Matteo explicitly licenses varying compound lengths. Run separately:

- `K2`: exactly 2 first syllables;
- `K3`: exactly 3;
- `K5`: exactly 5;
- `KMIX`: choose 2, 3, or 5 with equal probability for each generated type.

No regime is selected post hoc as the historical answer. All are reported.

## Output sampling

For comparability to the RF type inventory:

- generate 7,893 unique artificial surface types per regime where capacity permits;
- deterministic PRNG seed namespace based on `20260812` and regime;
- sampling with replacement at the source-word level; duplicate surface strings are discarded until the unique-type target is reached;
- record attempts/collision burden.

Primary simulation uses uniform unique source words. Robustness uses uniform unique first syllables.

## Frozen metrics

For each generated type set:

1. mean/median/p10/p90 character length;
2. character entropy;
3. `H(char | absolute position from left)`;
4. `H(char | position from right)`;
5. first-character entropy;
6. last-character entropy;
7. first-order `H(next char | previous char)`;
8. observed bigram-type count;
9. exact Levenshtein-distance-1 pair count;
10. mean/median/max edit-1 degree;
11. isolated-type fraction;
12. edit-1 location split: prefix / internal / suffix.

Voynich comparison uses the already frozen RF type inventory (7,893 exact-letter types); occurrence-weighted metrics are secondary context only because the historical artificial-word systems do not provide token-frequency weights.

## Controls

For the K2 primary regime, run two deterministic iid character controls with 7,893 unique strings and the same mean character length to rounding:

- uniform over the Latin-source alphabet;
- source-character-marginal iid.

These controls are not allowed to alter the historical generator.

## Decision interpretation

A literal Matteo surface mechanism earns `PARTIAL TRANSFER` only if it produces at least one nontrivial Voynich structural invariant beyond what the iid controls produce, while all substantial preregistered mismatches remain explicit.

It earns `STRUCTURAL TRANSFER` only if the main morphology, positional asymmetry, and local-dependency properties jointly transfer without tuning. A match in length alone or edit density alone is insufficient.