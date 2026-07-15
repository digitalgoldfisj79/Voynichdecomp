# Recoverability frontier v0.5.2 — long-text failure diagnosis

Date: 2026-07-15

Verdict: **SEARCH FAILURE CONFIRMED; LANGUAGE-MODEL OBJECTIVE RETAINED**

No Voynich text was scored.

## Diagnostic design

The 20 English and 20 Hebrew 384-character trials from the failed full generalisation were rerun with:

1. the existing flexible-inventory search;
2. a search supplied with the exact observed homophone-label multiset;
3. direct scoring of the true key under the frozen quadgram objective.

The language model, cipher data and `700,000 × 50` schedule were unchanged.

## English

Job: `Digitalgoldfish79/6a5809f4b1669a49bf076376`

Scientific SHA-256: `453967c414143fab8cdfc1c7bdfb8acbcf4912fbe58e0d537fbb6c44fa0576e1`

- flexible-inventory recovery: **9.2318%**;
- oracle-inventory recovery: **15.4297%**;
- true key outscored flexible recovered key: **20/20**;
- true key outscored oracle-inventory recovered key: **20/20**;
- mean true-key advantage over flexible result: **1.8290 score units per character**;
- mean true-key advantage over oracle-inventory result: **1.6854 per character**.

The exact inventory therefore does not solve the English failure. The search fails to reach a vastly better-scoring region of key space.

## Hebrew

Job: `Digitalgoldfish79/6a5809fd85d9643ce16d58e0`

Scientific SHA-256: `e39091d0651b2dacc78de2eaf03e5cbb7e8c43d57cd49f7b48639ff8af59e658`

- flexible-inventory recovery: **67.8776%**;
- oracle-inventory recovery: **94.6484%**;
- median oracle-inventory recovery: **100%**;
- true key outscored flexible recovered key: **17/20**;
- true key outscored oracle-inventory result: **15/20**.

Hebrew is principally an inventory-search problem on a minority of trials. With the correct inventory, recovery is nearly complete.

## Mechanistic diagnosis

The annealing schedule uses a fixed absolute temperature, while the language-model objective is a sum over sequence positions. Typical score differences therefore scale approximately with ciphertext length.

A temperature calibrated at 96 characters is effectively:

- twice as cold at 192 characters;
- four times as cold at 384 characters.

This can freeze the search in poor local optima even though the true key has a much better objective value. Strong frequency initialisation masked the defect in German, Finnish, Turkish and Arabic; English exposed it catastrophically.

## Required correction

Scale the annealing and reheating temperatures by effective scored length relative to the 96-character reference. Keep:

- the quadgram objective;
- all inventory constraints and moves;
- the `700,000 × 50` schedule;
- train/dev/test corpus boundaries.

The correction must be selected and checked on development data, then evaluated on a new untouched test block. The failed test block cannot be reused as validation.
