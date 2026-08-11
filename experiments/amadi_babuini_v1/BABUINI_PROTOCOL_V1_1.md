# Amadi Babuini v1.1 — source-corrected prospective protocol

Date: 2026-08-11
Branch: `experiment/amadi-babuini-v1-20260811`
Status: supersedes the unscored v1 61-unit design before any Voynich holdout was opened.

## Why v1 was superseded

The first engineering smoke showed that treating the 61 distinct signs observed in one solved babuini ciphertext as the complete historical codebook was wrong. Scheers' commentary says 61 different symbols were **used in that ciphertext** (14 single letters + 47 babuini); Amadi's underlying core table is larger. The source ledger describes section 0074 as a complete CV-syllable -> unique-symbol table, and later solved tables again show all CV syllables plus separately transposed single letters.

No Voynich target holdout was touched under v1. Its source-capacity failure is retained as an aborted-model result, not evidence against Babuini.

## Primary family: BAB_CORE_CV

This arm implements the operational architecture supported by sections 0074 and 0277–0278:

- ordinary plaintext letters have individual cipher signs;
- every consonant+vowel syllable has its own distinct cipher sign;
- `qua/que/...` are represented as q-u-vowel syllable units, reflecting the source note that q-syllables can be three letters;
- plaintext is segmented deterministically left-to-right: q-u-V first, otherwise C-V, otherwise a single letter.

Under the frozen 19-letter historical normalization this produces 19 single-letter units + 70 CV/qV units = 89 distinct plaintext units. This count is a normalization consequence, not a claim that Amadi's physical table contained exactly 89 cells.

## Voynich sign-unitisation

The frozen RF core19 representation remains unchanged. A literal one-RF-character = one-babuino model cannot express the source architecture at useful scale, so the only admitted compound-sign rule is prospectively fixed:

1. retain the 19 RF singleton signs;
2. on FIT-A only, rank adjacent within-word pairs by `count * max(PMI,0)`;
3. take the top 70 pairs;
4. tokenize greedily left-to-right, pair first when admitted, otherwise singleton.

This yields exactly 89 surface sign types, matching the normalized core Babuini unit inventory. The extractor uses no language score and no held-out target data.

Synthetic controls are rendered as 19 singleton base signs + 70 unique two-base-sign compounds. The extractor is not given the true sign boundaries; end-to-end recovery therefore tests both unitisation and decipherment.

## Language model and solver

A word-sensitive trigram model is trained over deterministic Babuini units from the Italian training corpus. The cipher is a document-global 89x89 substitution between surface signs and Babuini units.

Frozen optimiser after engineering smoke:

- 26,000 proposals per restart;
- up to 12 restarts per A/B ensemble;
- batches of four;
- convergence requires weighted A/B map agreement >= 0.95 and score difference <= 1e-7 nats/Babuini-unit.

## Qualification

Before target use:

- 12 fresh positive controls with disjoint fit/holdout spans;
- median held-out Babuini-unit recovery >= 0.95;
- all positive fits must converge;
- absolute score floor = 5th percentile positive-control holdout score;
- 60 structured negatives: 12 each iid, bounded Markov, motif-repeat, copy, slot;
- at most 2/60 negatives may reach the positive floor.

Any failure => `CALIBRATION_BLOCKED`; no target score.

## Target split

Previously opened Amadi H2 is never reused. The still-sealed Amadi C2 is split prospectively under namespace `AMADIBABUINICOREV1::<folio>` into `BAB_H1` and still-sealed `BAB_C1`. The exact manifest is frozen before target scoring.

FIT-A is inherited unchanged from the prior programme. Pair selection is performed on FIT-A only and then frozen.

## Decision

- converged + H1 score below frozen floor -> `CLOSED_NEGATIVE`;
- nonconverged -> `UNRESOLVED_SEARCH` regardless of score;
- converged + H1 score at/above floor -> `BAB_H1_CANDIDATE`;
- only a candidate may justify a separately frozen source-narrowing step before `BAB_C1` is opened.

No plaintext inspection and no optimiser change after BAB_H1.

## Expanded 1,365-combination syllabary

Sections 386–389 are retained as a separate broader residual. They are not silently approximated by BAB_CORE_CV. If the core arm closes, the 1,365-combination architecture remains open until separately unitised and qualified; if the core arm produces a candidate, expansion is not permitted as post-hoc rescue or improvement before confirmation.
