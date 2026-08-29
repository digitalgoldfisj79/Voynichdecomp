# WLCP v0.1 — FINAL

## RETRACTED FINDINGS

1. The first-pass claim of an independent adjacent/Markov **length-transition grammar** is retracted/narrowed. After conditioning on Currier, manuscript section, normalized line position and line-length bin, the metric does not resolve this: full EVA effect = -0.000645 bits/transition; matched-null SD = 0.001637; z = -0.39. Raw-character representation also does not resolve this: effect = -0.002743; matched-null SD = 0.001716; z = -1.60.
2. The first extension-family z-scores are retracted because that audit used a uniform feasible-cell draw instead of the correct conditional hypergeometric null.
3. The corrected exact-margin extension-family z-scores are superseded for final inference because lexical pairs can share token types and physical lines. Final inference uses disjoint lexical families plus a within-physical-line I/F swap null.

## ENDPOINT

**WL-1: genuine word-length/word-form positional organisation, cryptanalytically non-discriminating.**

The robust surviving result is a positional rule inside mechanically related word families: the longer member of a one-edge-extension pair is preferentially line-initial rather than line-final.

Final line-clustered/disjoint-family results (5,000 null permutations):

- Full EVA: effect = 0.558862 log-odds; matched-null SD = 0.081641; z = 6.85; OR = 1.748; empirical p = 0.000200.
- Currier A EVA: effect = 0.349398; matched-null SD = 0.133267; z = 2.62; OR = 1.420; p = 0.009598.
- Currier B EVA: effect = 0.538511; matched-null SD = 0.104454; z = 5.16; OR = 1.716; p = 0.000200.
- Full raw-character: effect = 0.566454; matched-null SD = 0.080750; z = 7.01; OR = 1.766.
- Currier A raw-character: effect = 0.504783; matched-null SD = 0.135796; z = 3.72; OR = 1.660.
- Currier B raw-character: effect = 0.522733; matched-null SD = 0.105322; z = 4.96; OR = 1.689.

The final null swaps the two observed edge tokens within each physical line with probability 0.5. This preserves line composition, section, Currier, token frequencies, token lengths and line-level clustering. Lexical extension pairs are disjoint, so a token type cannot contribute to multiple selected families. The comparison is stratified by family and manuscript section.

The raw first/final length difference is not universal: full EVA effect = 0.428620 units; matched-null SD = 0.037495; z = 11.43; Currier A does not resolve it (effect = 0.075763; matched-null SD = 0.067606; z = 1.12), while Currier B is strong (effect = 0.630333; matched-null SD = 0.048053; z = 13.12). The cross-Currier result promoted above is therefore the within-family positional rule, not a generic claim that all initial words are longer.

## Identifiability bound

Whole-token length cannot in principle distinguish identity/no cipher from a boundary-preserving one-symbol-per-plaintext-symbol substitution: both produce exactly the same token-length sequence for every plaintext. Any statistic solely of that sequence therefore has maximum possible difference exactly zero between those mechanisms.

Gate 3 consequently fails. No WL-2 mechanism exclusion and no WL-4 plaintext-length recovery claim is promoted. Existing Terminal/SVT mechanisms are not silently modified with post-hoc plaintext-word-boundary -> ciphertext-word-boundary rules.

## Audit

Circularity -> leakage -> confounds -> matched nulls -> control fairness -> measurement degeneracy -> representation dependence -> decision-rule fragility -> audit completeness -> interpretation were checked in that order. Later tests are explicitly bounding/post-hoc audits rather than being presented as preregistered discoveries.

Primary source SHA-256: `bf5b6d4ac1e3a51b1847a9c388318d609020441ccd56984c901c32b09beccafc`.
Parsed running text: 33,161 tokens; 4,097 lines; 207 folios; Currier A 10,301 tokens; Currier B 22,352; unclassified 508.

## Interpretation

Word length does not provide a new direct route to plaintext or identify a cipher family. It exposes a robust structural rule: **line position selects among related short/long Voynich word forms**. A future cipher-level claim requires a frozen mechanism that generates token boundaries and related short/long families and predicts this positional odds structure out of sample against matched non-cipher generators.
