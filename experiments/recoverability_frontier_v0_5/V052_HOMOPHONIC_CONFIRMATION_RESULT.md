# Recoverability frontier v0.5.2 — homophonic confirmation result

Date: 2026-07-15

Verdict: **PASS SMOKE CONFIRMATION; PROCEED TO SIX-LANGUAGE GENERALISATION**

No Voynich text was scored.

## Frozen solver

- arbitrary surface labels removed by first-occurrence canonicalisation;
- flexible bounded homophone inventory;
- smoothed train-only character quadgram objective with unigram term;
- deterministic global simulated annealing, reheating and greedy polishing;
- development-selected schedule: `700,000 × 50`;
- ciphertext length: 96 normalized characters;
- confirmation test chunks: untouched replicates 32–51.

## Confirmation results

| Language | Trials | Mean recovery | Median recovery | Exact recovery | Baseline | Job | SHA-256 |
|---|---:|---:|---:|---:|---:|---|---|
| English | 20 | 76.0938% | 92.1875% | 15.0% | 21.5625% | `Digitalgoldfish79/6a5806b1b1669a49bf07633b` | `1b6b7f4305f41d430fd64930c686d9163a7669f0c14b9845d5914be7aaa41d8b` |
| Turkish | 20 | 70.3646% | 80.7292% | 0.0% | 20.4167% | `Digitalgoldfish79/6a5806bbb1669a49bf07633d` | `47da99ccd19df666fcadf9410f304c81da8585785b162a827a3bc62e6459ffe8` |

Combined:

- mean recovery: **73.2292%**;
- baseline: **20.9896%**;
- exact recovery: **7.5%**;
- mean final homophone-inventory overlap: **89.4173%**.

Both languages exceed the frozen 60% floor and the combined result exceeds 70%.

## Interpretation

The initial homophonic failure was not solved by more trigram search. It required:

1. permitting the homophone inventory to change under a bounded multiplicity constraint;
2. replacing the trigram objective with a stronger quadgram objective;
3. retaining deep global search.

Performance remains substantially below monoalphabetic recovery at 96 characters and varies sharply across individual texts. A six-language and length-stratified test is therefore required before null-bearing homophonic development.
