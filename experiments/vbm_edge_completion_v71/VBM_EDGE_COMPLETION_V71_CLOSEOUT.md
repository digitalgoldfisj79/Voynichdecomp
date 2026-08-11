# VBM v7.1 — held-out surface-pair completion closeout

Date: 2026-08-11
Branch: `experiment/vbm-edge-completion-v71-20260811`
Namespace: `VBMEDGECOMP71`

## Binding disposition

**CLOSED AT DEVELOPMENT SMOKE. NO FORMAL CAL/VAL. NO VOYNICH TARGET ACCESS.**

HF smoke job: `6a7ba01727caad61c6eac7b5`.

The held-out pair-completion instrument successfully distinguished reusable-key ciphertext from fresh-key ciphertext, but it failed the registered stable surface-process adversaries. The failure exposes an observational-equivalence limit rather than a repairable threshold problem.

## Smoke results

Median held-out masked-cell advantage of the selected Bavarian/German latent factorisation over the strongest registered non-language completion baseline:

- BAV_GLOBAL: **+0.401796**
- GER_GLOBAL: **+0.364273**
- BAV_GLOBAL_SWAP: **+0.228788**
- BAV_FRESH: **−0.075638**
- GER_FRESH: **−0.062757**
- MARKOV1: **+0.447107**
- MARKOV2: **+0.771059**
- MARKOV3: **+0.345401**
- SLOT5: **+0.055794**

The reusable-key positives behaved as intended and selected their correct latent language in smoke. Fresh-key language controls were clearly rejected.

However, the stable Markov adversaries were not rejected. MARKOV1 exceeded BAV_GLOBAL and GER_GLOBAL; MARKOV2 exceeded every reusable-key positive by a large margin; MARKOV3 overlapped the positive range.

## Why this is structural

The synthetic positive family is a memoryless homophonic emission system with disjoint surface-sign pools driven by a first-order latent language Markov chain. Because each emitted surface sign identifies its latent source state in this construction, the resulting observed surface process is itself representable as a first-order Markov chain.

A surface Markov generator fitted to that induced law can therefore reproduce the same pair matrix and the same masked-pair completion structure without containing an independently recoverable plaintext/cipher mechanism.

Consequently:

1. exact-bigram-preserving v7 cannot distinguish the cipher because it preserves nearly all of its likelihood;
2. pair-completion v7.1 cannot distinguish it because a surface Markov process can reproduce the same pair law;
3. no statistic depending only on the observed first-order pair distribution can identify 'cipher' versus 'surface Markov grammar' for this cipher class.

This is an identifiability result about the experiment class, not evidence against all ciphers.

## What remains scientifically testable

The live question must now be formulated in terms of **parsimony / constrained generative description**, external historical constraints, or evidence outside the first-order surface distribution.

A legitimate successor could test whether a historically grounded latent cipher model gives a substantially shorter out-of-sample/prequential description length than unrestricted surface models after model-complexity costs are charged. That would test whether the latent explanation is a better compression/explanatory model, not whether it is uniquely identifiable from the pair law.

Alternatively, a genuinely new historical cipher operation that creates observables not reducible to a first-order surface Markov process could reopen direct cryptanalytic discrimination.

## Data disposition

- Formal v7.1 CAL: not run.
- Formal v7.1 VAL: not run.
- Voynich FIT: not run.
- H1/C1: not reused.
- No plaintext or mapping inspected.
