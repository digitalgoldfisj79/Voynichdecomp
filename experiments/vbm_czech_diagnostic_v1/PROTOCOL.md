# VBM Czech diagnostic v1 — bounded exploratory extension

Date: 2026-08-12
Namespace: `VBMCZECHDIAGV1`
Base: closed VBM v6.1 branch `experiment/vbm-key-transfer-v61-20260811`.

## Status and scope

This is a bounded language-extension diagnostic, not a reopening of the closed VBM cipher programme. The VBM programme remains limited by the established stable-surface/latent-transducer identifiability problem. Any Voynich result here is exploratory.

The only scientific addition is a Czech plaintext language model. Cipher topology, 19-state normalized plaintext alphabet, 21 core / 123 bridge surface geometry, homophone machinery, FREQ_PROP+CYCLE key family, moment+EM fitting machinery, surface-null family, and the deterministic sixfold Voynich FIT split are inherited unchanged.

Czech source is pinned to Universal Dependencies `UD_Czech-CAC` commit `798f89716ae5a96e86042df7d394d56787e2e213`, using train for LM fitting and dev+test for synthetic plaintext controls. Text is passed through the frozen VBM normalization (`unidecode`, j→i, v/w→u, y→i, x/z→s, retain `abcdefghilmnopqrstu`). Therefore this tests Czech under the existing VBM representation, not an orthographically rich or Old-Czech-specific model.

## Q0 — synthetic recognisability

Before interpreting the target, run two registered same-key Czech controls using the existing v4 discriminative harness:

1. `UNIFORM + FLAT`;
2. `FREQ_PROP + CYCLE`.

Use the inherited v4 settings: 18,000 fit events, 7,000 held-out events, max 40 EM iterations; competing latent LMs are Bavarian, German and Czech; the registered strong nonlanguage comparison is `best_null`, which selects among iid, hierarchical Markov, typed hierarchical Markov, periodic/slot and typed periodic/slot models.

A Q0 control qualifies only under the inherited v4 rule: Czech wins A, B and mean held-out HMM score; mean margin ≥ 0.02 nats/event; Czech paired-fit score gap ≤ 0.10; and Czech HMM score exceeds the best registered surface null. Q0 passes only if both controls qualify.

If Q0 fails, stop: do not score Voynich FIT.

## Q1 — exploratory Voynich FIT comparison

If Q0 passes, use the exact v6 deterministic sixfold ordering of the already-consumed FIT corpus. In each fold, fit on five sixths and evaluate on the held-out sixth.

For every fold compute:
- Bavarian latent-HMM held-out score;
- German latent-HMM held-out score;
- Czech latent-HMM held-out score;
- best registered surface-null held-out score;
- winning latent language;
- winner-vs-surface-null delta;
- Czech-vs-surface-null delta;
- Czech-vs-best(Bavarian,German) delta.

Primary interpretation:
- Relative Czech rank alone is not plaintext evidence.
- A Czech advantage is only interesting if it also beats the registered surface null out of sample.
- Because FIT is already consumed and VBM is closed on identifiability, even a positive result cannot establish Czech plaintext or a cipher.

No H1, C1, decoding, symbol-to-plaintext mapping, key interpretation, parameter tuning, new cipher family, or follow-on experiment is permitted in this diagnostic.

## Stopping rule

Close after Q0 or Q1. Do not chase a failed or ambiguous result.
