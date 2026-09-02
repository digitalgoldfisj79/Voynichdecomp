# VBM v10 — final one-shot compact recovery test

Date: 2026-09-02
Status: **FROZEN BEFORE THIS BINDING RUN**

## Purpose

This run answers one terminal falsification question only:

> Given the correct plaintext language, the exact admissible value inventory, a true single global VBM codebook, and 2,000 synthetic lines, can the frozen strong GPU solver recover the codebook and untouched plaintext reliably?

A failure here terminates the present whole-nucleus VBM programme. A pass does not validate VBM; it only permits the separate full-tail stress test.

## Why one size

The terminal question is recoverability at the maximum favourable Stage-A evidence level. The 2,000-line corpus is the largest compact Stage-A corpus and remains below the usable Voynich line count. Smaller phase-transition sizes are diagnostic, not needed to falsify practical recoverability at maximum evidence.

## Data and architecture

Unchanged from `VBM_V10_TERMINAL_IDENTIFIABILITY_PROTOCOL.md`:

- source-faithful `N0,B0,N1,...,B(k-1),Nk` architecture;
- 30 bridge surface types, six homophones per vowel;
- 96 nucleus surface types, three homophones per each of 32 consonant-run values;
- one global codebook per replicate;
- three German and three Italian independent replicates;
- deterministic modern `wordfreq` banks used instrumentally only;
- correct language and exact candidate inventory supplied to the solver;
- first 1,600 lines FIT, final 400 lines untouched HOLDOUT;
- no Voynich plaintext is used.

## Binding solver

Unchanged from `VBM_V10_GPU_SEARCH_ADDENDUM_V2.md`:

- 4× A100 execution;
- eight deterministic independent chains;
- exact character-5-gram FIT likelihood;
- bridge blocks of 8: exhaustive `5^8 = 390,625` assignments;
- nucleus blocks of 4: exhaustive `32^4 = 1,048,576` assignments;
- three fixed partitions: descending-frequency contiguous, frequency-interleaved, deterministic shuffled;
- exact single-site polish before and after block sweeps;
- no truth-key information enters search.

The only oracle condition fitted in this final test is `O2_TRUE_LANGUAGE_ZERO_KEY`. `O0_TRUE_KEY` is scored as a sanity reference but is not fitted. O1 is omitted because it cannot rescue an O2 failure and is not part of the terminal criterion.

## Software qualification

The already frozen GPU/CPU exact-likelihood smoke is executed inside the same paid job before binding data. The job aborts immediately if the scorer disagrees with CPU by more than `1e-5` or if the coordinate-polish smoke fails. This is not a separate paid job.

## Primary untouched-HOLDOUT statistics

For each of six replicates:

- `REC_CHAR` — exact decoded-character recovery;
- `REC_B` — occurrence-weighted bridge-value recovery;
- `REC_N` — occurrence-weighted non-empty nucleus-value recovery;
- `REC_CHAR5`, `REC_B5`, `REC_N5` — same statistics for surface types seen at least five times in FIT;
- `HOLD_LM`;
- `RAND_HOLD_LM`, `RAND_HOLD_SD`, and `HOLD_ADV` against 20 deterministic random dictionaries;
- FIT/HOLDOUT type coverage;
- all eight chain FIT scores and total exact conditional assignments evaluated.

## Frozen terminal criterion

The compact architecture survives only if all are true at 2,000 lines:

1. at least 5/6 replicates have `REC_CHAR >= 0.80`;
2. at least 5/6 have both `REC_B >= 0.70` and `REC_N >= 0.70`;
3. each language passes at least 2/3 replicates on both conditions;
4. at least 5/6 have `REC_CHAR5 >= 0.90`, `REC_B5 >= 0.80`, and `REC_N5 >= 0.80`.

If any conjunct fails, verdict:

`VBM_GLOBAL_KEY_NOT_RECOVERABLE_EVEN_COMPACT`

and no full-tail or Voynich run is permitted.

If all four pass, verdict:

`VBM_COMPACT_RECOVERY_GATE_PASSED_FULL_TAIL_REQUIRED`

This is survival, not evidence for VBM.

## Compute budget

Hardware: HF `a100x4` (4× A100 80 GB), currently $10/hour.
Hard job timeout: 45 minutes, therefore maximum hardware charge $7.50. The job is expected to finish substantially sooner from the measured exact-likelihood throughput. No additional large job is automatically launched.
