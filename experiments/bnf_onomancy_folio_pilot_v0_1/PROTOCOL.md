# BnF 7342 Numerical-Alphabet Voynich Folio Pilot v0.1

Status: FROZEN BEFORE LANGUAGE-SCORING RESULTS
Date: 2026-08-08
Branch: experiment/bnf-onomancy-folio-pilot-v0.1-20260808

## Aim
Test whether a numerical/homophonic substitution class derived from the five parallel letter-value tables transcribed from BnF lat. 7342 can generate an objectively language-like decoding of one Voynich folio. This is a falsification-oriented pilot, not a claim that BnF 7342 documents encryption or that Voynich uses this mechanism.

## Voynich data
Source: repository root `voynich_transcriptions_slim.json` (blob SHA observed before run: fd4b28ade3bf2b00ef80bb616a03d2894508a2ac).
Independent transcription families fixed for the pilot: ZLZI, TTLI, PCCA, FFSG, GCGA, VDRB, RGVN.

Folio selection was mechanical and performed before any language scoring: among folios represented in all seven families, choose the folio whose median token count is closest to the median token count of all fully covered folios; break ties lexicographically. This selected **f10r**, with per-family token counts 88, 91, 87, 91, 86, 85, 87 (median 87).

## BnF-derived alphabet
Plain alphabet: `abcdefghiklmnopqrstuxyz` (23 letters; no separate j/v/w).

Five transcribed value systems are denoted F (first), M (material), G (Greek), L (Latin), H (Hebrew). The aggressive all-safe-pair construction yields the following maximum homophone capacities per plaintext letter, counting only table-pair/value combinations unique within their own table pair:

`a7 b7 c7 d10 e10 f8 g10 h10 i10 k9 l9 m9 n9 o7 p10 q9 r9 s6 t9 u6 x8 y10 z10` (total 199).

The conservative four globally injective table pairs F+M, F+G, F+L, M+L give capacity 4 for every letter (total 92).

## Cipher/rendering hypotheses
Run separately:

- G1: Voynich transliteration characters are ciphertext symbols; monoalphabetic capacity 1 per plaintext letter where symbol count permits.
- G2: Voynich transliteration characters are ciphertext symbols; conservative BnF homophonic capacity 4.
- G3: Voynich transliteration characters are ciphertext symbols; aggressive BnF safe-pair capacities above.
- T2: Voynich word-types are ciphertext symbols; conservative capacity 4. Spaces are not interpreted as plaintext word boundaries; each Voynich token decodes to one plaintext character.
- T3: Voynich word-types are ciphertext symbols; aggressive capacities above.

No claim is allowed to transfer between these rendering hypotheses.

## Language panel
Frozen before decoding: Latin, Italian, German, French, Greek, Hebrew, Arabic, Castilian Spanish.

External language models use Universal Dependencies treebanks as reproducible public corpora, preferring historical corpora where readily available (Latin ITTB; Old French SRCMF if fetchable) and otherwise standard UD corpora. Greek/Hebrew/Arabic are deterministically romanized with Unidecode for this pilot. All languages are normalized to the same 23-letter alphabet: accents stripped; j->i; v->u; w->u; nonalphabetic material removed except spaces in glyph-mode models.

This normalization is a pilot convention and is not evidence for historical transliteration practice.

## Scoring
Character 4-gram language models with additive smoothing. Search is simulated annealing / random-restart assignment under the model-specific homophone capacities. Search seed is derived deterministically from 20260808, transcription, language and model.

For glyph modes Voynich spaces are retained as fixed word boundaries. For token modes spaces are removed because each token is treated as one ciphertext symbol.

## Nulls
For the representative ZLZI transcription, each model/language combination is compared with order-shuffled null ciphertexts preserving the ciphertext symbol multiset; glyph-mode shuffles preserve the positions of spaces. Each null is decoded with the identical optimizer and capacity constraints. Report standardized separation of the real-folio optimum from the null distribution.

## Predeclared interpretation
A language is `PILOT_SIGNAL` only if:
1. it ranks first for the same rendering model in at least 5 of the 7 transcription families; and
2. on ZLZI its real-v-null z score is >= 3.0 for that model; and
3. the result is not dependent solely on the most permissive T3 model. A T3-only result is `FLEXIBILITY_ONLY` regardless of score.

Anything else is `NONRESOLVING` or `NEGATIVE` for this pilot. Generated plaintext strings are diagnostic outputs, not translations.
