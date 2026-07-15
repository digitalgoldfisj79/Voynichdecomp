# Recoverability frontier v0.5.1 — monoalphabetic locked-test result

Date: 2026-07-15

Verdict: **PASS STAGE 1A; PROCEED TO HOMOPHONIC SOLVER DEVELOPMENT**

No Voynich text was scored.

## Frozen design

- Branch: `experiment/recoverability-frontier-v0.5.1-solver-portfolio-20260715`
- Input: first-occurrence recurrence-canonicalised ciphertext symbols
- Source model: language-specific smoothed character trigram plus unigram term, trained on corpus `train` only
- Search: deterministic simulated annealing with global restarts, reheating and greedy polishing
- Development-selected schedule: `700,000` iterations × `50` restarts
- Test: 6 languages × 3 lengths × 20 unseen source chunks and unseen keys = **360 trials**
- Noise: 0% in Stage 1A
- No test-time schedule selection or parameter tuning

The development schedule was selected because it achieved 93.1876% mean character recovery and raised the weakest language, Arabic, to 85.2105%.

## Aggregate locked-test result

- mean normalized character recovery: **94.6166%**;
- frequency-ranking baseline: **34.9826%**;
- exact plaintext recovery: **29.7222%**;
- weakest language: Hebrew at **88.7240%**;
- every language exceeded the frozen 50% floor;
- overall mean exceeded the frozen 70% gate.

| Language | Mean recovery | Frequency baseline | Exact recovery | Shard job |
|---|---:|---:|---:|---|
| English | 98.1988% | 33.9974% | 40.00% | `Digitalgoldfish79/6a58015e85d9643ce16d5799` |
| German | 98.3030% | 40.2387% | 30.00% | `Digitalgoldfish79/6a580166b1669a49bf0761c2` |
| Finnish | 96.3411% | 30.5035% | 48.33% | `Digitalgoldfish79/6a58016db1669a49bf0761ca` |
| Turkish | 94.5356% | 29.9349% | 38.33% | `Digitalgoldfish79/6a580175b1669a49bf0761ce` |
| Hebrew | 88.7240% | 34.9609% | 10.00% | `Digitalgoldfish79/6a58017db1669a49bf0761d0` |
| Arabic | 91.5972% | 40.2604% | 11.67% | `Digitalgoldfish79/6a580185b1669a49bf0761d2` |

## Results by ciphertext length

| Normalized characters | Mean recovery | Frequency baseline | Exact recovery |
|---:|---:|---:|---:|
| 96 | 87.3958% | 29.3663% | 8.33% |
| 192 | 98.1380% | 35.8594% | 33.33% |
| 384 | 98.3160% | 39.7222% | 47.50% |

The length profile is substantial. Near-complete recovery is routine from 192–384 characters. At 96 characters the solver remains useful but exact recovery is uncommon, particularly for Hebrew, Arabic and Turkish.

## Frozen gate

```json
{
  "mean_accuracy_pass": true,
  "language_floor_pass": true,
  "pass": true
}
```

## Shard provenance

| Language | Scientific SHA-256 |
|---|---|
| English | `32a5016efc79c6251136a72469ce2cf7494a78d34d76743ca7ec3ec36ff16a69` |
| German | `4a8512f778b25e43c7f345107a7db22bffc1af690bb990ebd7097adea5aeef8d` |
| Finnish | `b98e36144bbe13d9f8cc94956da646d57574e945444573b7dfbe56ee7ee5578e` |
| Turkish | `af3e46f4eca1a461f44f015f330b3c4c173cf560736985939323047477e53f08` |
| Hebrew | `3accb5c092fa18fc8cdb01eb6e8b2f6cf2b55cea3e3f23522a5568f182e43489` |
| Arabic | `441d1bfe6c0005cb7781edf367813c462c7472bacc52d8fd3c88dc716feb489d` |

Aggregate summary SHA-256: `e4b962598aeaec048edfb7bad572a8427409d8185698e2463162d784db988501`

Complete gzip/base64 JSON artifacts are preserved in the immutable logs of each shard job.

## Interpretation

The v0.5.0 monolithic Transformer failure was not evidence that fresh-key synthetic cryptanalysis is generally inaccessible. A family-specific, key-invariant solver recovers monoalphabetic plaintext strongly across typologically and script-diverse languages.

This establishes:

- the corpus split and language-model layer are usable;
- arbitrary surface-label invariance can be preserved;
- explicit cipher-family structure is critical;
- search quality, rather than generic neural capacity, was the principal monoalphabetic bottleneck;
- short texts remain materially harder and must be reported separately.

It does not establish:

- that an unknown text is monoalphabetically enciphered;
- that cipher can yet be distinguished from structured generation;
- that other cipher families are recoverable;
- that the Voynich Manuscript should be tested.

## Next frozen target

Proceed to family-specific solvers for:

1. homophonic substitution;
2. homophonic substitution with nulls.

The next stage must preserve arbitrary symbol-label invariance and evaluate latent plaintext recovery. It must not reinstate a direct `MESSAGE` classifier. Broad blind-family selection remains deferred until multiple family-specific solvers pass independently.
