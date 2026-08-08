# BnF 7342 Numerical-Alphabet Voynich Folio Pilot v0.1 — Result

Date: 2026-08-08
Branch: `experiment/bnf-onomancy-folio-pilot-v0.1-20260808`
Protocol freeze commit: `715868c0d66063db2450e70a0a00b491f6dbde07`
Original runner commit: `c615b94c03ca35569e8d582e995791fe40457490`
Computational-only amendment: `627d28d8d564d13b2e6464273792f3b3938fd343`
Amended runner commit: `9c138a5342d2465befa6127a6936db2f0de10f65`
Completed HF job: `6a77a3233e1f34a7e32bf63b` (225 s running)
Cancelled pre-result implementation job: `6a77a29ada2af92a634efb7f`

## Verdict

**NONRESOLVING / NO PILOT SIGNAL.**

The BnF-derived numerical/homophonic substitution class did not identify a stable underlying language on mechanically selected Voynich folio f10r under the preregistered criteria.

## Mechanical folio selection

All seven fixed transcription families covered f10r. Token counts were ZLZI 88, TTLI 91, PCCA 87, FFSG 91, GCGA 86, VDRB 85, RGVN 87; median 87. Selection used coverage and median-length proximity only, before language scoring.

## Predeclared signal table

| model | consensus top language | families agreeing | ZLZI z for that language | verdict |
|---|---|---:|---:|---|
| G1 glyph monoalphabetic | Spanish | 2/7 | 13.397 | NONRESOLVING |
| G2 glyph conservative homophonic | German | 2/7 | 14.473 | NONRESOLVING |
| G3 glyph aggressive safe-pair | Italian | 2/7 | 13.114 | NONRESOLVING |
| T2 token-type conservative homophonic | Spanish | 4/7 | 1.556 | NONRESOLVING |
| T3 token-type aggressive safe-pair | Latin | 3/7 | 1.123 | NONRESOLVING |

No model satisfies the preregistered requirement of a single language ranking first in >=5/7 transcription families with ZLZI z>=3, and T3 alone could never qualify as more than FLEXIBILITY_ONLY.

## What the very large glyph-model z scores mean

They are **not language evidence**. G1/G2/G3 obtain very large true-v-shuffled separations for mutually incompatible languages. The shuffled null destroys Voynich local/within-token structure while retaining only symbol frequencies and space positions; a high z therefore shows that the real glyph stream has exploitable local structure, which is already known, not that a particular plaintext language has been recovered.

The winning language is unstable across transcription systems:
- G1: no language exceeds 2/7 families.
- G2: no language exceeds 2/7.
- G3: no language exceeds 2/7.

Generated strings are visibly optimizer-shaped pseudo-language (e.g. repeated `aca/cca/caca`-like structures) rather than translations.

## The more relevant homophonic token models

Treating Voynich word-types as homophones is much closer to the proposed many-code-symbols-per-plaintext-letter mechanism. Here the apparent signal collapses toward the null:
- T2 ZLZI best = Spanish, z=1.556.
- T3 ZLZI best = Latin, z=1.123.
- Cross-transcriber T2 gives Spanish top in 4/7, still below the frozen 5/7 criterion and with weak representative z.
- T3 gives Latin top in 3/7 and is the most permissive model in any case.

Thus the model class does not recover a stable Latin, Italian, German, French, Greek, Hebrew, Arabic or Castilian plaintext on f10r.

## Strong diagnostic sanity check

A post-result hash audit showed PCCA and RGVN are **byte-identical on f10r**:
`5418ce4e2022709e1435a5990eca4d8cc76ec8cf5f02d702cc767f6022d2fc8c`.

Despite identical ciphertext, stochastic runs selected different winning languages in several glyph models (e.g. G2 PCCA -> Latin, RGVN -> Hebrew; G3 PCCA -> Italian, RGVN -> German). Therefore the identity of the top language in those models is demonstrably sensitive to optimizer/null realization. This further invalidates any attempt to read the large glyph z scores as linguistic identification.

## Corpora actually used

- Latin: UD Latin-ITTB (22,775 sentences; ~2.35M raw chars)
- Italian: UD Italian-ISDT
- German: UD German-GSD
- French: UD French-GSD (the preregistered Old French URL was unavailable and the frozen fallback fired)
- Greek: UD Ancient Greek-Perseus
- Hebrew: UD Hebrew-HTB
- Arabic: UD Arabic-PADT
- Spanish: UD Spanish-AnCora

Greek/Hebrew/Arabic were deterministically romanized with Unidecode and every corpus normalized to the 23-letter BnF alphabet. These are pilot comparators, not claims about historical transliteration practice.

## Interpretation

This experiment does **not** falsify the mathematical fact that the five BnF tables can be adapted into a reversible or homophonic cipher. It does show that the straightforward applications tested here do not explain f10r in a way that selects a stable underlying language.

The strongest positive-looking effect (glyph-v-shuffle) is nonspecific; the mechanism-specific token/homophone effect is weak and transcription-unstable. The correct result is therefore negative/nonresolving, not a candidate decode.
