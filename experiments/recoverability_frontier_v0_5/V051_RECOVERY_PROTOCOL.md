# Recoverability frontier v0.5.1 — frozen recovery-only protocol

Date: 2026-07-15

Status: **development pilot; no Voynich inference permitted**

## Scientific correction

v0.5.0 incorrectly trained a classifier to distinguish held-out corpus text from generated latent text after both had been passed through identical cipher channels. That distinction is not identifiable from ciphertext. v0.5.1 removes messagehood classification from the recoverability task.

Every enciphered sequence has a known latent target in synthetic calibration, including sequences emitted by structured generators. The primary question is now:

> How accurately can the latent sequence be recovered under unseen text, key, parameter, language, cipher family, length and noise conditions?

## Frozen corpus and cipher boundary

v0.5.1 reuses without modification:

- the six SHA-256-pinned Universal Dependencies corpora;
- train/dev/test corpus partitions;
- the eight v0.5.0 cipher implementations;
- fresh independent keys by sample;
- lengths 96, 192 and 384;
- noise levels 0%, 1% and 3%;
- first-occurrence recurrence canonicalisation;
- deterministic non-overlapping source chunks within each corpus partition.

No test plaintext, key or parameter draw may be used in training or development.

## Evaluation populations

### Natural-source population

Held-out corpus chunks from the fixed test partitions.

### Generated-source stress population

Latent sequences produced by the four frozen structured generators, then passed through the same cipher and noise channels. These sequences now retain their generated latent targets and are scored for exact recovery. They are not labelled `NO_MESSAGE`.

The generated population tests whether a solver is recovering ciphertext structure or merely hallucinating fluent language.

## Arm A: recovery-only recurrence Transformer

Two independent models are trained from scratch:

1. family-known: receives language and cipher-family tags;
2. blind-family: receives only the language tag.

Training uses natural-source positives only. The classifier head is excluded from the loss and from evaluation. Every test ciphertext is decoded.

Primary neural metrics:

- normalized character accuracy;
- exact sequence recovery;
- family-wise and language-wise accuracy;
- accuracy by length and noise;
- natural versus generated-source accuracy;
- teacher-forced token negative log-likelihood;
- sequence confidence and calibration against actual recovery.

## Arm B: family-specialist solver portfolio

A single generic neural model does not define cryptanalytic recoverability. v0.5.1 therefore develops specialist solvers, beginning with:

- monoalphabetic substitution: simulated annealing / hill climbing with train-only character n-gram scoring;
- homophonic substitution: many-to-one key search with split/merge proposals;
- transposition: block-size and permutation search with n-gram scoring;
- polyalphabetic: period estimation and per-column key search.

Later specialist additions cover nulls, nomenclators, feedback and fractionation. Each solver must be selected and tuned on development data only.

## Confidence and abstention

Abstention is based on predicted recovery reliability, not alleged messagehood.

A confidence mapping is fitted on development examples from solver-internal scores to observed character accuracy. Test output is one of:

- `RECOVERED`: predicted accuracy at least 70%;
- `PARTIAL`: predicted accuracy from 30% to below 70%;
- `LOW_RECOVERY_CONFIDENCE`: predicted accuracy below 30%;
- `UNSUPPORTED_FAMILY`: no validated solver applies.

Thresholds are fixed before test evaluation.

## Primary gates

### Neural baseline gate

For each arm:

- report all test cases without classifier filtering;
- mean natural-source character accuracy at least 50% in at least three materially distinct cipher families;
- exact recovery reported but not required for initial continuation;
- no claim of broad recovery if any result is driven by one language or one length regime.

### Specialist portfolio gate

For a cipher family to be called recoverable:

- development-selected solver mean test accuracy at least 70%;
- at least 50% mean accuracy at every tested length in the noiseless condition;
- unseen-key performance retained;
- calibrated `RECOVERED` predictions have empirical mean accuracy at least 70%;
- generated-source stress cases do not receive high confidence unless their actual latent sequences are recovered.

### Stop rules

Stop or redesign a family if:

- oracle recovery is high but all independent solvers remain below 30%;
- apparent performance comes only from teacher forcing or language fluency;
- confidence is uncalibrated on held-out keys;
- test-informed threshold or solver selection occurs;
- row-level result preservation fails.

## Result preservation

Every full job must either:

- write to a durable authenticated bucket; or
- emit gzip+base64 row-level JSON through immutable job logs using the committed artifact wrapper.

Aggregate-only logs are insufficient for formal validation.

## Scientific boundary

Passing v0.5.1 establishes recoverability under the tested language, cipher, solver and noise families. It does not distinguish generated text from enciphered text in an unknown manuscript and does not prove that the Voynich Manuscript is a cipher.
