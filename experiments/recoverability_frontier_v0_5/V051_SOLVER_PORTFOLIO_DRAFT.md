# Recoverability frontier v0.5.1 — solver-portfolio draft

Date: 2026-07-15

Status: **future protocol; does not modify or rescue the frozen v0.5.0 test**

## Why this layer is necessary

A failure of one generic encoder-decoder does not establish that plaintext information is unrecoverable. The recoverability frontier must compare a preregistered portfolio containing both general and family-specialist cryptanalytic methods. Model selection must occur on development data only, and all solvers must face identical held-out keys, texts, lengths, noise and controls.

The present recurrence canonicalisation follows the key-invariant principle used by Kambhatla, Born and Sarkar, *Decipherment as Regression: Solving Historical Substitution Ciphers by Learning Symbol Recurrence Relations* (2023):
https://consensus.app/papers/decipherment-as-regression-solving-historical-kambhatla-born/f1e8606950265cf78376c4eb32e495fd/

The portfolio should also include search-plus-language-model baselines because whole-sequence neural language-model scoring and beam search have improved substitution decipherment relative to local n-gram scoring:
https://consensus.app/papers/decipherment-of-substitution-ciphers-with-neural-kambhatla-bigvand/d176481780a45a2292da9a0809e50374/

For historical deployment, source models must be period- and genre-matched where possible; historical language models have materially improved homophonic decipherment in English and German:
https://consensus.app/papers/historical-language-models-in-cryptanalysis-case-studies-megyesi-sikora/cf183e8991295edd833c62889feb9187/

Segmentation must be a separate inference stage for variable-length numerical or glyph streams:
https://consensus.app/papers/segmenting-numerical-substitution-ciphers-aldarrab-may/fa8b97fa8ef553c28a1ffdda6324d5a9/

## Portfolio architecture

### Shared input and scoring

- first-occurrence recurrence encoding and raw positional features;
- language-specific character n-gram models trained on corpus `train` only;
- neural character language models trained from scratch on `train` only;
- development-only solver and threshold selection;
- normalized Levenshtein character accuracy on test positives;
- exact recovery, key recovery where defined, and message/no-message calibration;
- clustered uncertainty by underlying source chunk.

### 1. Monoalphabetic substitution

Run three independent solvers:

1. recurrence Transformer;
2. simulated annealing / hill climbing over bijective keys with character n-gram scoring;
3. beam search with whole-sequence neural-language-model scoring.

Freeze a development-selected ensemble rule before test.

### 2. Homophonic and null-bearing substitution

- recurrence Transformer;
- many-to-one key search with split/merge proposals;
- beam search with n-gram and neural-LM rest costs;
- explicit null-state proposals and null-rate prior;
- oracle-segmentation and inferred-segmentation arms reported separately.

### 3. Nomenclator

- lattice containing character substitutions and candidate whole-word code expansions;
- language-model beam search over the lattice;
- alternating optimisation of character key, codeword lexicon and null assignments;
- strict held-out word-code and key testing.

### 4. Polyalphabetic systems

- period candidates from recurrence/autocorrelation and repeated-pattern evidence;
- per-column shift/key search followed by joint language-model reranking;
- beam or particle search for non-additive alphabet mappings;
- test periods excluded from development ranges.

### 5. Stateful feedback systems

- bounded state/key enumeration where feasible;
- Viterbi or beam decoding over latent state;
- particle search for larger state spaces;
- separate reporting for substitution noise versus insertion/deletion desynchronisation.

### 6. Transposition

- block-size hypotheses;
- simulated annealing, hill climbing and tabu-style permutation proposals;
- joint substitution-plus-transposition alternating search;
- language-model scoring of reconstructed order;
- explicit incomplete-block handling.

### 7. Fractionated systems

- phase and coordinate-pair boundary hypotheses;
- row/column symbol partition inference;
- alternating coordinate-map and plaintext-language optimisation;
- insertion/deletion-aware pair realignment before key search.

### 8. Blind-family routing

The blind system does not use one monolithic decoder alone. It produces calibrated probabilities over mechanism families and sends each sample to the top-k preregistered solvers. Candidate plaintexts are compared under a common held-out scoring and complexity-penalty framework. It must return `NO_MESSAGE` or `NON_IDENTIFIABLE` when no candidate exceeds the development-frozen evidence threshold.

## Selection rule

For each family, the primary v0.5.1 result is the performance of a development-frozen portfolio rule, not the post-hoc maximum test score. Individual solver results remain mandatory. A family is called recoverable only when:

- at least one solver exceeds the preregistered accuracy threshold on test;
- the development-selected portfolio retains performance on unseen keys and parameters;
- matched controls remain below the false-positive gate;
- performance is not driven by one language or length regime.

## Compute strategy

- CPU arrays for n-gram construction and independent stochastic-search restarts;
- one job per language × cipher family for embarrassingly parallel search;
- GPU batching for recurrence Transformers and neural-language-model scoring;
- A100/L40S only where measured throughput justifies it;
- aggressive early stopping on development bounds;
- no broad expansion until the six-language pilot passes.

## Boundary

v0.5.1 would estimate the best demonstrated recoverability under the tested portfolio. It still would not establish universal impossibility for failed families, nor would synthetic recoverability alone classify the Voynich Manuscript.
