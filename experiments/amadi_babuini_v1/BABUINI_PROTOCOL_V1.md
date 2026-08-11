# Amadi Babuini v1 — prospective protocol

Date: 2026-08-11
Branch: `experiment/amadi-babuini-v1-20260811`
Parent: completed `experiment/amadi-residuals-v1-20260811` closeout.

## Scope

This programme tests the Amadi **babuini** residual that was not directly tested by Amadi Residuals v1. It is a new prospective algorithm. Previously opened Amadi H2 is not reused for target inference.

Source facts carried into the test:

- the solved first babuini example is reported as 14 ordinary letters plus 47 babuini signs = 61 cipher signs;
- babuini signs encode syllabic material and produce strong compression (about 2.5 cipher signs for about six plaintext letters in the analysed example);
- Amadi sections 386–389 describe a much larger 1,365-combination syllabary spanning VC, VCC, VCCC, CV, CCV, CCCV and CVC structures.

The first two facts motivate the exact 61-unit cardinality. The full-syllable source motivates the second, broader inventory. Neither source licenses arbitrary post-target glyph grouping.

## Families

### B61-CV

A 61-unit plaintext inventory consisting of:

- 14 single plaintext letters, chosen prospectively as the 14 most frequent letters in the Italian training corpus;
- 47 most frequent source-permitted CV/VC syllables in the same training corpus.

### B61-FULL

The same 14 single letters plus the 47 most frequent source-permitted syllables matching one of:

`CV, VC, VCC, VCCC, CCV, CCCV, CVC`.

This is a bounded stress-test of the 1,365-combination system, not a claim that the 47 selected corpus units reproduce Amadi's exact historical codebook.

## Plaintext segmentation gate

Each natural-language word is segmented deterministically by dynamic programming into the fewest allowed units; ties prefer the longer current unit, then lower frozen unit id. Whole words that cannot be represented are excluded. Both Italian training and disjoint control corpora must retain >= 99.5% of plaintext characters. Otherwise the family is `SURFACE_INCOMPATIBLE_PLAINTEXT_COVERAGE` and never reaches Voynich.

## Cipher-sign unitisation

The Voynich surface remains the frozen RF core19 representation from Cipher Coverage v1, with the exact inherited RF SHA-256.

To test the historically attested 61-sign scale without arbitrary target-driven grouping, the only admitted compound-sign extraction is:

1. retain the 19 RF singleton signs;
2. on FIT-A only, count within-word adjacent RF pairs;
3. rank pairs by `count * max(PMI,0)` with deterministic tie-breaking;
4. take the top 42 pairs;
5. tokenize greedily left-to-right, taking an admitted pair when available, otherwise a singleton.

This yields a fixed 19+42 = 61 sign inventory. The pair list is frozen from FIT-A and then applied unchanged to held-out target text.

The same extraction rule must qualify end-to-end on fresh synthetic babuini controls. Synthetic 61-sign ciphertext is rendered prospectively as 19 singleton base glyphs plus 42 unique digraph codewords over the same 19-symbol base alphabet. The pair extractor is not given the true boundaries.

## Solver

After unitisation, Babuini is treated as a global 61x61 monoalphabetic substitution between cipher signs and the frozen 61 plaintext units. A word-sensitive trigram language model is trained over the plaintext units. Optimisation uses two independent simulated-annealing ensembles with the budget frozen before formal qualification:

- 24,000 proposals per restart;
- up to 12 restarts per ensemble;
- evaluated in batches of four;
- convergence requires A/B score difference <= 1e-7 nats/unit and frequency-weighted map agreement >= 0.95.

Smoke budgets are engineering-only and have no scientific status.

## Qualification

For each family:

- Q1: 12 fresh Italian positive controls, each with disjoint fit and holdout spans; median unit recovery >= 0.95 and every fit converged.
- Q2/specificity: 60 structured negatives, 12 each from iid, bounded Markov, motif-repeat, copy, and slot generators.
- Absolute held-out positive floor: 5th percentile of the 12 positive-control holdout scores.
- Specificity gate: at most 2/60 structured negatives may reach the positive floor.

A family failing qualification is `CALIBRATION_BLOCKED` and never sees target holdout.

## New target holdout

The old Amadi H2 is considered contaminated for this new algorithm and is not used.

The still-sealed 23-folio C2 from Amadi Residuals v1 is deterministically split by SHA-256 namespace `AMADIBABUINIV1::<folio>` into:

- `BAB_H1`: first half after hash sort;
- `BAB_C1`: second half after hash sort.

The split and exact folio lists are frozen before target scoring. FIT-A remains the inherited old fitting set.

`BAB_C1` stays sealed unless a fully qualified family both converges on FIT-A and reaches its frozen absolute positive-control floor on `BAB_H1`.

## Decision rule

- converged + below absolute floor -> `CLOSED_NEGATIVE`;
- nonconverged target fit -> `UNRESOLVED_SEARCH`, regardless of score;
- converged + at/above floor -> `BAB_H1_CANDIDATE`, allowing a separate exact-source narrowing step before opening `BAB_C1`;
- no plaintext inspection or optimizer change after BAB_H1.

## Stop rule

The programme stops immediately if both families are structurally incompatible or calibration-blocked. It stops at BAB_H1 if neither family is a candidate. BAB_C1 must not be opened merely because a decoded string looks interesting.
