# v0.6 Family T — bounded columnar transposition

Date: 2026-07-16

Status: **FROZEN BEFORE IMPLEMENTATION OR RESULTS**

No test data or Voynich text has been inspected.

## Scope

Family T covers the historically material non-block transposition gap:

- global ragged columnar transposition;
- line-reset ragged columnar transposition;
- each optionally composed with one fresh monoalphabetic substitution.

It does **not** include arbitrary spirals, geometric routes or manuscript-specific readout paths. Adding such routes without independent historical evidence would create an unconstrained spearfishing problem. Fixed local permutations were already tested in v0.5.5.

## Literature provenance

The search architecture must reproduce the GPU/metaheuristic columnar-transposition approach described by Dimitrov & Esslinger, *CUDA Tutorial — Cryptanalysis of Classical Ciphers Using Modern GPUs and CUDA* (2021), arXiv:2103.13937, before any custom amendment is considered.

## Generator

- plaintext source length: 384 characters;
- observed line boundaries generated independently of transposition mode;
- development widths: 4–10;
- locked-test widths: 11–14;
- fresh non-identity column order per trial;
- ragged columns: no padding symbol or exposed rectangle dimensions;
- spaces and punctuation-normalised boundaries are transposed as ordinary characters;
- optional fresh monoalphabetic substitution applied before transposition;
- final first-occurrence recurrence representation retained as a blind control, not as the sole input.

## T1 — true substitution key; unknown transposition

The solver must select:

- global versus line-reset mode;
- width;
- column order;
- plaintext ordering.

Gate across 16 English development trials:

- mean plaintext recovery at least 95%;
- minimum at least 85%;
- mode accuracy at least 14/16;
- width accuracy at least 14/16;
- exact canonical column order at least 12/16.

## T2 — true transposition; unknown substitution key

Use the validated v0.5.1 monoalphabetic solver after oracle detransposition.

Gate:

- mean plaintext recovery at least 95%;
- minimum at least 90%;
- at least 14/16 trials at or above 95%.

## T3 — fully blind joint recovery

Permitted only if T1 and T2 both pass.

Initial architecture:

1. GPU or CPU-parallel column-order metaheuristic for each mode-width candidate;
2. alternating validated monoalphabetic refinement;
3. full mode-width model comparison under a fixed complexity penalty;
4. explicit abstention when candidate scores are not separated.

Development gate:

- mean plaintext recovery at least 80%;
- median at least 90%;
- at least 14/16 trials at or above 80%;
- mode accuracy at least 14/16;
- width accuracy at least 13/16;
- no catastrophic trial below 40%.

One development-only amendment is permitted. The generator, corpus, gates and test split cannot change.

## Locked test

A passing solver is frozen and evaluated once on 20 untouched test trials using widths 11–14. No post-test modification is permitted. Voynich application is forbidden unless the locked test passes the same recovery and abstention criteria.