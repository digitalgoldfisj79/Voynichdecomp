# BnF 7342 homophonic-cipher positive-control qualification v0.2

Date: 2026-08-08
Parent: `experiment/bnf-onomancy-folio-pilot-v0.1-20260808`
Status: **FROZEN BEFORE DECODER RESULTS**

## Question
Can the v0.1 language-guided decoder recover known plaintext language and substantial plaintext content after text is encrypted with the exact BnF-7342-derived homophonic constructions at Voynich-f10r-like sample size?

This is an instrument qualification. It makes no Voynich historical inference.

## BnF alphabet and tables
Alphabet (23 letters): `a b c d e f g h i k l m n o p q r s t u x y z`.

Tables, in alphabet order:
- F: `1,2,3,4,5,6,7,8,9,10,10,2,12,22,4,12,24,6,16,4,20,8,24`
- M: `1,2,3,4,5,28,10,12,1,16,2,12,23,6,2,20,3,30,9,1,20,0,4`
- G: `1,2,6,4,5,8,1,6,7,1,8,8,5,6,5,2,2,1,4,1,1,3,3`
- L: `1,2,6,4,1,8,4,3,10,2,3,8,5,6,8,7,2,6,1,6,5,0,7`
- H: `1,2,6,4,5,6,3,1,3,6,2,4,1,6,7,2,8,6,1,6,1,0,7`

The four globally injective pairs are frozen as `FM, FG, FL, ML`.
Across all ten table-pairs, locally unique pair-values produce exactly 199 safe marked code symbols, with per-letter capacities:
`a7 b7 c7 d10 e10 f8 g10 h10 i10 k9 l9 m9 n9 o7 p10 q9 r9 s6 t9 u6 x8 y10 z10`.

## Cipher controls
Only the mechanism-specific token/homophone models from v0.1 are qualified here.

- **T2:** each plaintext letter independently chooses one of its four codes from the globally injective pairs FM/FG/FL/ML.
- **T3:** each plaintext letter independently chooses one of all locally safe marked codes from the 199-code construction.

The marked BnF code symbols are then renamed to opaque IDs before decoding. The decoder therefore knows only the capacity class, not the true table pair or number values. This matches the v0.1 Voynich token-type hypothesis.

An oracle decoder using the retained generation codebook must reproduce plaintext exactly; otherwise the run is invalid.

## Normalization
All plaintext and LM text is deterministically transliterated/normalized to the 23-letter alphabet:
- Unicode romanization with `Unidecode` where needed;
- lowercase;
- `j -> i`, `v -> u`, `w -> u`;
- retain only the 23 BnF letters;
- word spaces removed for T2/T3 scoring, matching v0.1.

For the Middle High German Primer's ASCII bracket notation, bracketed vowel/consonant codes are reduced deterministically to their base letter sequence before the common normalization.

## Language panel
The decoder must rank the correct language against the same eight-way family used in v0.1:
Latin, Italian, German, French, Greek, Hebrew, Arabic, Spanish.

LM corpora:
- Latin: UD Latin-ITTB, with a deterministic 80/20 sentence split; LM uses only train 80%.
- Italian: UD Italian-ISDT, 80/20 split.
- German: UD German-GSD, 80/20 split.
- Hebrew: UD Hebrew-HTB, 80/20 split.
- French: UD French-GSD.
- Greek: UD Ancient Greek-Perseus.
- Arabic: UD Arabic-PADT.
- Spanish: UD Spanish-AnCora.

No held-out control span may enter its language model.

## Positive-control tiers
### P0: favorable in-domain heldout
For Latin, Italian, German, Hebrew, sample plaintext only from the held-out 20% of that language's LM corpus.

### P1: historical/out-of-domain
- Latin: UD Latin-LLCT independent of ITTB.
- Italian: UD Italian-Old (Dante).
- German: the Middle High German Nibelungen-Lied selection in Joseph Wright's *A Middle High German Primer* (Project Gutenberg 22636), specifically the `B. Text. [A]VENTIURE XVII` extract, not the modern-German Gutenberg Nibelungen edition.
- Hebrew: Sefaria export, `Mishneh Torah, Torah Study`, Hebrew, Torat Emet 363.

P1 is evaluated only as historical/domain-robustness evidence; P0 is the binding instrument floor.

## Sample sizes and replicates
Lengths in normalized plaintext characters: **88, 176, 352**.
88 is the binding f10r-like size because f10r has about 85-91 tokens across the fixed transcription panel.

For each target language, tier, length and cipher model, use two deterministic plaintext/encryption replicates (`r0`, `r1`).

## Decoder
Capacity-constrained homophonic substitution decoder derived from v0.1:
- character 4-gram LM, no spaces;
- simulated annealing / hill-climb;
- 4 restarts;
- 3500 proposals per restart;
- deterministic seeds derived from SHA-256 of the ciphertext + model + candidate language + restart, so identical ciphertext cannot yield different language rankings merely because it was given a different transcription label.

## Nulls
For each encrypted control, run four independently optimized global token-order shuffles. Shuffles preserve ciphertext symbol counts exactly while destroying plaintext sequential order.

Report target-language z relative to these nulls, but exact recovery metrics below are primary because plaintext is known.

## Metrics
For every case:
1. oracle plaintext language rank before encryption;
2. blind decoder language rank among 8 languages;
3. exact decoded character accuracy against known normalized plaintext;
4. code-symbol mapping accuracy on observed symbols;
5. target-language true-v-shuffle z;
6. unique ciphertext-symbol count.

## Binding qualification gates at length 88
A model is qualified for a Voynich-f10r-like test only if **P0** satisfies all of:
- **Q0 language-model sanity:** correct language ranks #1 on >= 14/16 unencrypted P0 cases (4 languages x 2 reps x 2 models; model duplication counts only for bookkeeping and must agree because plaintext is identical).
- **Q1 blind language recovery:** correct language ranks #1 on >= 12/16 encrypted cases, and no target language is below 2/4.
- **Q2 plaintext recovery:** median exact character accuracy >= 0.50 across the 16 encrypted cases.
- **Q3 null separation:** median target-language z >= 3.0.

If any P0 gate fails at 88, verdict is **NOT QUALIFIED AT VOYNICH-FOLIO SCALE**. Longer lengths are then a power curve only.

If all P0 gates pass but P1 fails the same thresholds, verdict is **QUALIFIED IN-DOMAIN / HISTORICALLY DOMAIN-LIMITED**.
If both P0 and P1 pass, verdict is **QUALIFIED FOR THIS MECHANISM CLASS**.

T3 is never evidence for the historical use of the BnF tables by itself; this test only qualifies or disqualifies the decoder.

## Prohibited interpretation
A positive control pass does not make the earlier f10r negative a historical falsification of the BnF mechanism. It only establishes that the cryptanalytic instrument has power under the specified synthetic conditions. A failure means the earlier f10r result is instrument-nonresolving for this mechanism class.