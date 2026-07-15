# Recoverability frontier v0.5.2 — homophonic solver diagnostic

Date: 2026-07-15

Status: **frozen development diagnostic; no Voynich inference permitted**

## Motivation

v0.5.1 passed the monoalphabetic locked test at 94.6166% mean character recovery across six languages and unseen keys. Stage 1B tests whether the same explicit language-model search can recover a bounded many-to-one homophonic substitution.

## Cipher family

The existing `homophonic` generator is unchanged:

- each plaintext character owns one or more cipher symbols;
- symbol multiplicity is frequency-adaptive;
- each occurrence samples uniformly among that character's symbols;
- the surface symbol labels are randomly permuted for every trial;
- no nulls or channel noise are included in v0.5.2.

Null-bearing homophonic substitution is deferred until ordinary homophonic recovery passes.

## Input invariance

Cipher symbols are canonicalised by order of first occurrence. Raw synthetic symbol integers are never supplied to the solver.

## Family-known structural assumption

The solver knows the bounded multiplicity rule used by the benchmark family, but not the key. Because short ciphertexts may not exhibit every potential homophone, the observed key inventory is constructed by selecting the most probable homophone slots implied by the train-corpus unigram distribution. The selected slot multiset has exactly the number of distinct observed cipher symbols.

This is a development-stage bounded-family assumption. Later validation must vary and hide the multiplicity rule.

## Language model and search

- smoothed character trigram plus unigram term;
- fitted on corpus `train` only;
- repeated plaintext labels are assigned to observed cipher symbols;
- deterministic simulated annealing swaps assignments among cipher symbols;
- global shuffled restarts, reheating and greedy polishing are retained from the passing v0.5.1 mono solver;
- schedule selection uses `dev` only.

## Development smoke

- languages: English and Turkish;
- length: 96 normalized characters;
- unseen source chunks and keys;
- at least two search schedules;
- frequency-slot assignment is the baseline.

## Full diagnostic if smoke passes

- languages: English, German, Finnish, Turkish, Hebrew and Arabic;
- lengths: 96, 192 and 384;
- development replicates: 8 per language × length;
- locked test replicates: 20 per language × length;
- one schedule selected on development and applied once to test.

## Primary metrics

- normalized character recovery;
- exact recovery;
- baseline recovery;
- results by language and length;
- runtime per ciphertext.

## Gate

Proceed to null-homophonic development only if the locked homophonic test achieves:

- at least 70% overall mean character recovery;
- at least 50% mean recovery in every language;
- at least 60% mean recovery at 96 characters;
- no test-time schedule or inventory adjustment.

## Scientific boundary

Passing this diagnostic establishes recoverability only for this bounded synthetic homophonic family. It does not distinguish cipher from generation, validate an unknown homophone inventory, or justify a Voynich run.
