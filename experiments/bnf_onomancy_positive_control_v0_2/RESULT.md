# BnF 7342 homophonic-cipher positive-control qualification v0.2 — Result

Date: 2026-08-08
Branch: `experiment/bnf-onomancy-positive-control-v0.2-20260808`
Protocol freeze: `65453591ec43c7b0b09a9c74136d172906496f58`
Runner: `d9514655eaa6944bdc95df7a2d0e4e7b4e79fdca`
HF job: `6a77a844da2af92a634efbaa` — COMPLETED, 419 s running
Cases: 96

## Verdict

**NOT QUALIFIED AT VOYNICH-FOLIO SCALE.**

The v0.1 language-guided homophonic decoder cannot reliably recover known plaintext encrypted with the exact BnF-derived T2/T3 constructions at the 88-character scale corresponding to approximately one f10r token stream. Therefore the earlier negative/nonresolving f10r experiment cannot be upgraded into evidence against the BnF mechanism class. The instrument itself lacks demonstrated power.

## Binding P0 gate at length 88

| Gate | Frozen requirement | Result | Status |
|---|---|---:|---|
| Q0 plaintext LM sanity | correct language >=14/16 | **16/16** | PASS |
| Q1 blind language recovery | >=12/16; each language >=2/4 | **5/16**; Latin 1, Italian 0, German 1, Hebrew 3 | FAIL |
| Q2 exact plaintext recovery | median >=0.50 | **0.1023** | FAIL |
| Q3 true-v-shuffle separation | median z >=3.0 | **1.9991** | FAIL |

Thus the language models themselves identify all favorable unencrypted controls correctly, but the homophonic decipherment step does not recover the known plaintext.

### By mechanism at 88

| model | blind correct | median target-language character accuracy | median target z | median observed cipher types |
|---|---:|---:|---:|---:|
| T2 four globally injective BnF pairs (92 possible marked codes) | 3/8 | 0.1591 | 2.0742 | 45.0 |
| T3 all locally safe BnF pair-values (199 possible marked codes) | 2/8 | 0.1023 | 1.0595 | 61.5 |

## Power curve

Longer ciphertext does not rescue the current optimizer at the tested lengths.

### P0 favorable heldout controls

| length | model | blind correct / 8 | median target char acc | median z |
|---:|---|---:|---:|---:|
| 88 | T2 | 3 | 0.1591 | 2.0742 |
| 88 | T3 | 2 | 0.1023 | 1.0595 |
| 176 | T2 | 1 | 0.1307 | 1.6682 |
| 176 | T3 | 0 | 0.0966 | 0.1175 |
| 352 | T2 | 1 | 0.1776 | 3.3436 |
| 352 | T3 | 0 | 0.1335 | 0.7481 |

The important metric is `char_acc_target`: this evaluates the best mapping found while supplying the **correct** language model. Its low values show that cross-language score calibration is not the main failure; the mapping search / identifiability problem remains even under the correct language.

## Historical/out-of-domain tier P1

P1 also fails at 88:
- oracle language: 14/16 (passes Q0 threshold);
- blind correct: 3/16;
- median target character accuracy: 0.0852;
- median target z: 1.1149.

The P1 controls were:
- Latin: UD Latin-LLCT;
- Italian: UD Italian-Old (Dante);
- German: Middle High German Nibelungen-Lied selection from Joseph Wright, *A Middle High German Primer*, Project Gutenberg 22636, beginning `D[o] erbiten si der nahte`;
- Hebrew: Sefaria `Mishneh Torah, Torah Study`, Hebrew, Torat Emet 363.

P1 additionally exposes some domain-shift failures in the language panel: one 352-character Middle High German span ranked French before encryption, and one 88-character Mishneh Torah span ranked Spanish before encryption after the frozen romanization. These are secondary because P0 already fails decisively.

## What is and is not learned

1. The exact BnF codebook construction is internally valid: oracle inversion reproduced every generated plaintext exactly.
2. The ordinary eight-way language models are competent on the favorable unencrypted P0 controls (16/16 correct at 88).
3. The current blind capacity-constrained homophonic search is not competent at Voynich-folio scale.
4. The earlier f10r `NONRESOLVING` result must remain an instrument-nonresolution, not a negative test of the historical BnF mechanism.
5. T3 is harder than T2, as expected from its larger homophone inventory, but even T2 is far below the frozen recovery floor.

## Next admissible work

Do not test another Voynich folio with this decoder yet.

The next instrument programme should determine the recoverability frontier under known plaintext controls, separating three issues:
- **information length:** extend T2/T3 controls to 704, 1408, 2816+ characters;
- **search quality:** benchmark the current annealer against stronger homophonic-substitution solvers while the true mapping is known;
- **model calibration:** use per-language likelihood-ratio / heldout-calibrated scores rather than comparing raw n-gram log probabilities across independently trained/romanized language models.

Only after a solver recovers known 88-character controls at a preregistered rate should a one-folio Voynich test be considered resolving.