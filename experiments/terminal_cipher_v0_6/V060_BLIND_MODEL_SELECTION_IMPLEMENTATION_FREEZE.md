# v0.6 blind model-selection and Voynich representation freeze

Date: 2026-07-16

Status: **FROZEN BEFORE SYNTHETIC LOCKED RESULTS OR VOYNICH SCORING**

## Purpose

Family P passed its development and locked recovery gates. Passing recovery is necessary but not sufficient for manuscript application. This stage asks whether label-invariant statistics can distinguish P from other cipher, generated, notation and ordinary-text classes, with calibrated abstention, before any Voynich evidence is exposed.

## Synthetic classes

Nine mutually exclusive classes are fixed:

1. `mono`: fresh monoalphabetic substitution;
2. `P`: periodic or line-reset Alberti-style wheel substitution, periods 2–12 according to the existing train/dev/test split;
3. `S`: fresh homophonic, null-homophonic or fractionated substitution controls;
4. `T`: bounded substitution-plus-block-transposition controls;
5. `mixed`: Family P followed by a bounded block permutation;
6. `generated`: matched Markov-2, motif, copy-mutate and slot generators;
7. `notation`: a bounded slot/template notation generator with dominant separators;
8. `ordinary`: held-out ordinary language streams;
9. `none`: iid, shuffled and alternating-motif out-of-family controls.

The six pinned v0.5 languages are crossed with all classes. Train, calibration and locked-test examples use their corresponding corpus partitions and deterministic disjoint seeds. Sequence length is fixed at 384 symbols to match the validated Family P cell.

## Features

Only label-invariant features are permitted:

- normalized unigram, bigram and trigram entropy;
- conditional entropy and transition diversity;
- collision and recurrence-distance spectra;
- lagged equality and mutual information for lags 1–24;
- global and line-reset phase statistics for candidate periods 2–12;
- sorted phase-histogram agreement, which is invariant to symbol names;
- line-start distribution effects and line-length variation;
- n-gram type ratios;
- LZ78 phrase rate and zlib compression ratio;
- alphabet occupancy and ranked symbol masses.

No plaintext recovery score, language identity, true key, true period, section, hand or semantic feature is supplied to the classifier.

## Classifier and calibration

- classifier: deterministic ExtraTrees multiclass ensemble;
- calibration: one-vs-rest isotonic calibration fitted only on the calibration split and renormalized;
- abstention: a probability and top-two margin pair chosen on calibration data to maximize P recall subject to at most 5% P false positives on `generated`, `notation` and `none` controls.

## Locked synthetic gate

All conditions must pass:

- macro one-vs-rest AUC at least 0.90;
- multiclass expected calibration error at most 0.05;
- P false-positive rate on structured generated/notation/none controls at most 5%;
- P recall at least 0.80;
- P precision at least 0.90.

Failure blocks Voynich scoring. No post-test feature, classifier, calibration or threshold change is permitted.

## Frozen Voynich representations

The authoritative asset is `voynich_transcriptions_slim.json`, preserving IVTFF folio and line boundaries.

Three independently sourced EVA streams are fixed:

1. `ZLZI` — Zandbergen–Landini, primary;
2. `TTLI` — Takeshi Takahashi / LSI, robustness;
3. `VDRB-1` — VMS Database RF source, robustness.

A fourth required representation is `ZLZI-line-recurrence`, in which each line is independently canonicalized by first occurrence before concatenation. This preserves line-aware recurrence while removing glyph identity.

Normalized transcription text is used as supplied. Token boundaries are represented by one separator symbol. No uncertain reading is manually resolved, no glyph is merged or split after scoring, and daiin is not used as the authoritative stream.

Windows are line-aligned, non-overlapping 384-symbol blocks formed separately by section and deterministic folio fold. Windows never cross section or fold boundaries.

## Frozen positive Voynich selection rule

A Family P detection requires all of the following:

- `P` is the aggregate top class in each of ZLZI, TTLI and VDRB-1;
- at least 70% of windows in each raw EVA stream meet the frozen P probability and margin threshold;
- `P` is the aggregate top class in ZLZI-line-recurrence and at least 60% of its windows meet threshold;
- each of the two deterministic folio folds has at least 60% P-evidence windows in every raw EVA stream;
- at least four sections represented by at least eight windows each show a P majority in all three raw EVA streams.

Anything else is an abstention or out-of-family result. No isolated high-probability page or section is sufficient.

## Solver compatibility rule

The validated Family P plaintext solver assumes an independently observed circular ciphertext alphabet with the same cardinality as the candidate language alphabet. EVA glyph order is not an observed historical ring order. Therefore a positive blind P classification does not by itself authorize an English-looking decode.

Direct plaintext application is permitted only if, before decoding:

- symbol cardinality matches a locked-passing language model without padding, dropping or merging glyphs;
- circular symbol order is independently specified rather than optimized on Voynich output;
- the same mode/period structure transfers across the three EVA streams, both folio folds and held-out sections;
- independent restarts converge.

If these compatibility conditions are unavailable, the result remains structural P evidence or abstention, not a decipherment.
