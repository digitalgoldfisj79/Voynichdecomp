# Tranchedino × STA Historical Cipher Programme v2.3 — Stage B1 calibration

Date frozen: 2026-08-09
Namespace: `TRANCHSTA23B1`
Parent B0 result: `STAGE B0 PASS`
Status: **instrument development/qualification only — no Voynich target fit authorised**

## 1. Scientific question

Can a blind solver recover a fresh fifteenth-century Paduan plaintext encrypted with the **exact strict mixed-unit anatomy of Tranchedino f.69v**, when the visible symbols are opaque and plaintext word spaces are not supplied?

The binding key contains 92 surface signs:

- 36 alphabetic homophones for the frozen 19-letter alphabet;
- 11 geminate signs: `bb cc dd ff gg ll nn pp rr ss tt`;
- 7 null signs;
- 38 one-sign nomenclator entries;
- 0 syllabic signs.

This is the only primary model in B1. No section/page/hand-specific keys, no same-family ordering carrier, no syllabary and no target-derived token construction are admitted.

## 2. Why this is not the old v0.5.4 nomenclator experiment

The old v0.5.4 development family used a fresh **monoalphabetic** character key plus whole-word codes, retained encrypted spaces, initially tested about 384 plaintext characters, and failed principally because only 2–7 nomenclator symbols were observed in many chunks. B1 differs prospectively in four binding ways:

1. genuine f.69v **homophonic** alphabet geometry, not monoalphabetic substitution;
2. the historically attested f.69v geminate and null classes are present;
3. plaintext spaces are discarded from the cipher surface;
4. the source-only B0 census fixes 12,000-letter controls and a 38-from-top-96 codebook regime, exposing at least 26/38 code identities and at least 171 code occurrences in all twelve census controls.

Old code may be reused as implementation scaffolding, but old results cannot qualify this instrument.

## 3. Frozen source and language model

Historical source archive SHA-256:
`ddae949a2d4ff13714204f3751feaf9e836333ef57a45def77c803cd87fc7b61`.

Binding line source:
`paduan_lines.csv`, SHA-256
`c5eba63cbe8055d3506d099043f5df23fd427df709546df6de70e084fedd3cf6`.

Use the same 19-letter normalisation as Stage A:

`j→i`, `v/w→u`, `y→i`, `x/z→s`; retain only `abcdefghilmnopqrstu`.

Language-model training remains the old chronological partition before page 183:

- 4,119 lines;
- 172,347 retained characters.

Fit the same line-reset additive-alpha=0.5 character quadgram/unigram model used in Stage A. No modern corpus or pretrained language model enters the primary objective.

## 4. B1 source-panel split

The old held-out source contains 69 text-bearing pages at/after page 183 and 54,750 retained characters.

Sort those pages by SHA-256 of
`TRANCHSTA23B1source::<page>`.

- first 17 pages: **D1 development source** (~14,117 retained characters);
- remaining 52 pages: **Q1 qualification source** (~40,633 retained characters).

D1 may be inspected for solver development. Q1 truth and recovery metrics remain sealed until the B1 implementation is frozen.

The source split itself is fixed before any B1 solver result.

## 5. Fresh mixed-key generator

For every control:

### 5.1 Nomenclator

Take the 96 most frequent normalised training words, with no post-B0 filtering. Select 38 distinct words uniformly without replacement using the control seed. Assign one opaque nomenclator surface sign to each selected word. Every occurrence of a selected word is replaced by its sign.

### 5.2 Geminates

For every non-nomenclator word, scan left-to-right. If the next two letters are one of the eleven binding f.69v geminates, emit that geminate sign and consume both letters. Otherwise emit the appropriate alphabetic sign and consume one letter.

Geminate use is deterministic when eligible; no optional-use parameter is fitted.

### 5.3 Alphabetic homophones

Use the exact strict f.69v multiplicities:

- `a=1`, `c=1`;
- every other retained plaintext letter has two signs.

When a letter has two signs, choose uniformly and independently. No cyclic/sticky/stateful homophone schedule is added.

### 5.4 Nulls

After each substantive emitted sign, independently insert at most one null with probability `p_null`. If inserted, select one of the seven null signs uniformly.

The four qualification nuisance strata are frozen as:
`p_null ∈ {0.01, 0.03, 0.06, 0.10}`.

These are calibration nuisance levels, not a claim about historical null frequencies. D1 has one control at each level; Q1 has three fresh controls at each level.

### 5.5 Surface relabelling and boundaries

Randomly permute all 92 key labels before emission so numeric/lexical labels leak no class information. The solver knows K=92 and the class cardinalities but not the permutation, codebook or mappings.

Preserve line boundaries only. **Discard plaintext word spaces completely.**

The solver therefore receives a sequence of opaque one-sign cipher units plus hard line breaks.

## 6. Source-only expansion constraint

B0 source simulations showed that 12,000-letter Paduan controls under this mechanism have plaintext-character / cipher-event ratios approximately 0.93–1.10 over the four null strata. To prevent a variable-output solver from winning trivially by assigning frequent signs to empty strings or unusually long words, the binding search domain requires:

`0.90 <= decoded_characters / observed_cipher_events <= 1.12`.

This is a broad source-derived admissibility interval, not a fitted target parameter.

No other source-frequency prior is part of the binding language score. Surface-frequency matching may be used only for initialisation/search proposals, never as evidence in the final recognition statistic.

## 7. Model capacity / MDL accounting

For the fixed K92 model and top-96 candidate word pool, the discrete key-space description cost is recorded prospectively as:

`ln C(92,7) + ln C(85,11) + ln C(74,38)`
`+ ln(11!)`
`+ ln P(96,38)`
`+ ln(36! / 2!^17)`

= approximately **368.832 nats = 532.111 bits**.

This accounts for class allocation, geminate assignment, word-subset/word-sign assignment, and the f.69v alphabetic assignment. Null signs are semantically interchangeable. The four calibration null-rate strata contribute no target fit parameter.

All target reporting, if later authorised, must disclose this capacity. No hidden section/key-state parameters may be added.

## 8. Solver objective and outputs

The binding semantic output inventory comprises:

- 36 one-character alphabet outputs with frozen multiplicities;
- 11 fixed two-character geminate outputs;
- 7 empty-string null outputs;
- 38 distinct word outputs selected from the frozen top-96 candidate pool.

The primary search objective is the frozen Paduan line-reset character language score over the expanded plaintext, subject to the expansion-ratio constraint. A word-level model may be used only as an explicitly declared search proposal/initialisation aid; the final objective and recognition score remain character-model based.

Decoded words/text may be inspected on D1 because D1 is development data. Q1 decoded strings remain sealed; only truth-based numerical recovery metrics may be emitted automatically.

## 9. Development allowance

Exactly one development phase is authorised on the four D1 controls (one per null stratum).

During D1, solver architecture, search scheduling and compute budget may be changed using D1 only. The generator, source split, candidate pool, class geometry, language model, metric definitions and Q1 gates in this document may not be changed.

At most three solver implementation revisions are allowed. After the final D1 revision, archive its code hash and search schedule as `B1_IMPLEMENTATION_FREEZE.md` before generating any Q1 result.

If no D1 implementation can achieve median plaintext recovery >=0.80 across the four controls, stop with:
`B1 DEVELOPMENT INSTRUMENT FAILED`.

No Q1 run is permitted.

## 10. Q1 binding qualification

Generate twelve fresh Q1 controls, three per null stratum, with independent plaintext windows, fresh 38-word codebooks, fresh homophone choices and fresh 92-label permutations.

Each control is solved by two independent optimiser ensembles A/B.

### 10.1 Convergence

All 12 controls must satisfy:

- A/B final objective difference <= 1e-4 nats per decoded character;
- occurrence-weighted A/B semantic-map agreement >=0.85 over surface signs observed in the control.

Failure of any control gives `B1 INSTRUMENT NOT QUALIFIED`.

### 10.2 Recovery gates

Across all 12 controls:

- median expanded-plaintext character recovery >=0.90;
- minimum expanded-plaintext character recovery >=0.75;
- median occurrence-weighted true semantic-map recovery >=0.85;
- minimum occurrence-weighted semantic-map recovery >=0.70;
- median nomenclator occurrence-word recovery >=0.80;
- minimum nomenclator occurrence-word recovery >=0.60;
- median geminate occurrence recovery >=0.90;
- minimum geminate occurrence recovery >=0.70;
- for non-zero null strata, median null-occurrence classification F1 >=0.90 and minimum >=0.75.

Only observed signs/occurrences count toward component mapping metrics; unobserved key-sheet entries are not scored as failures. Expanded plaintext recovery is computed against the complete normalised plaintext after removing spaces, using normalised edit accuracy.

### 10.3 Fixed-map generalisation

For each Q1 control, the solver fits only the first 80% of cipher lines. Freeze that map and score the final 20% without refitting.

A held-out surface event is scorable if its surface sign occurred in the fit portion; other events are hard breaks. Required fixed-map held-out coverage: >=0.95 for every control.

Archive the 12 authentic held-out character-model scores. The **5th percentile of these twelve scores**, computed by NumPy's default linear quantile, becomes the prospective absolute historical-control floor for any later Voynich H23 adjudication.

No threshold is chosen from Voynich data.

## 11. Specificity controls

After the authentic Q1 fits are complete but before any Voynich target score, construct 24 matched negative controls from the Q1 surface geometry:

- 12 within-line order-shuffle controls preserving line lengths and symbol counts;
- 12 IID-symbol controls preserving each control's empirical surface unigram distribution and line lengths.

Run the same blind fit/fixed-map procedure.

Qualification additionally requires:

- authentic-vs-pooled-negative held-out-score ROC AUC >=0.95;
- at most 1/24 negative controls scores at or above the frozen authentic 5th-percentile floor.

If specificity fails, verdict: `B1 DETECTOR NOT SPECIFIC`; no Voynich score is admissible.

## 12. Label-invariance audit

For one deterministic Q1 control chosen by SHA-256 before solving, apply a second independent permutation of the 92 visible labels and rerun. Recovery metrics and final objective must agree to numerical tolerance (`<=1e-10` objective difference after canonical semantic alignment). This establishes that nominal STA codes cannot themselves drive recovery.

## 13. Advancement rule

Only if D1 passes, the implementation is frozen, all Q1 convergence/recovery gates pass, specificity passes, and label invariance passes is the instrument labelled:

`TRANCHEDINO-STA MIXED-UNIT INSTRUMENT QUALIFIED`.

Only that label authorises a separately frozen Voynich T23/H23/C23 protocol.

A B1 failure closes **this exact f.69v one-sign mixed-unit instrument**. It does not automatically close a separately audited syllabic/variable-sign historical key, connected-aaa segmentation, or other unitisation not present in f.69v.
