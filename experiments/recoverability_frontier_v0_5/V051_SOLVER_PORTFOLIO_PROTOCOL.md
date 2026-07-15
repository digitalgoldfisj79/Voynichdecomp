# Recoverability frontier v0.5.1 — family-specific solver protocol

Date: 2026-07-15

Status: **frozen development diagnostic; no Voynich inference permitted**

## Motivation

v0.5.0 showed that a shared fresh-key sequence-to-sequence Transformer did not learn general cryptanalysis. The separate `MESSAGE` classifier was also invalid because a generator-produced latent sequence passed through a cipher remains an encoded sequence.

v0.5.1 therefore measures cryptanalytic recovery only. Every ciphertext has a known latent target, whether its source is human text or a generator. Provenance classification is deferred to a later bounded MDL comparison.

## Stage-1 families

Three structurally distinct families are tested first:

1. `mono`: monoalphabetic substitution, solved by language-model scoring plus stochastic key search;
2. `polyalphabetic`: periodic additive substitution, solved by period search and coordinate optimisation of shifts;
3. `feedback`: plaintext-feedback substitution, solved by exhaustive bounded search over key and initial state.

These families are selected because each admits an explicit family-specific solver. Failure here would invalidate the scoring or search layer before harder homophonic, nomenclator, transposition and fractionated solvers are attempted.

## Corpora and splits

The six SHA-pinned corpora in `corpus_manifest_v050.json` are unchanged.

- language models and frequency initialisation: `train` only;
- solver schedule selection: `dev` only;
- final diagnostic: `test` only;
- source chunks do not cross corpus partitions;
- fresh independent key for every ciphertext;
- no pretrained model and no Voynich text.

## Input representation

Cipher symbols are canonicalised by order of first occurrence. This preserves equality and recurrence while removing arbitrary numeric key labels.

## Language model

Each language uses a smoothed character trigram model trained on its `train` partition. The score is total held-out log probability. No test text contributes to model estimation.

## Development grid

- languages: English, German, Finnish, Turkish, Hebrew, Arabic;
- lengths: 96, 192 and 384 normalized characters;
- noise: 0% only in Stage 1;
- replicates: 8 per language × family × length for development;
- test replicates: 20 per language × family × length;
- independent source chunks and keys.

## Solver selection

### Monoalphabetic

Development may select among a small frozen grid of annealing iterations and restart counts. Frequency-ranked initial keys are combined with random perturbations. Only development recovery may choose the schedule.

### Polyalphabetic

Candidate periods 2–10 are evaluated. For each period, shifts are updated by coordinate ascent under the language-model score. The highest-scoring candidate is returned.

### Feedback

All key and initial-state pairs in the language alphabet are evaluated exactly. The highest-scoring plaintext is returned.

## Primary metrics

- normalized character accuracy;
- exact recovery rate;
- median character accuracy;
- results by family, language and length;
- runtime per ciphertext;
- language-model score advantage over the best frequency baseline.

Character accuracy is one minus normalized Levenshtein distance, floored at zero.

## Stage-1 gate

Proceed to implement the remaining five family solvers only if:

- `mono` mean test character accuracy is at least 70%;
- `polyalphabetic` mean test character accuracy is at least 85%;
- `feedback` mean test character accuracy is at least 95%;
- every family has at least 50% mean accuracy in every language;
- no solver parameter is changed after test inspection.

If the gate fails, diagnose the language model or solver on oracle and known-key ablations before further scale-up.

## Scientific boundary

Passing v0.5.1 would establish only that explicit bounded solvers recover known synthetic plaintexts under these three cipher families. It would not distinguish cipher from generation and would not justify application to the Voynich Manuscript.
