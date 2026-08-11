# VBM v7 CRITE — smoke closeout

Date: 2026-08-11
Branch: `experiment/vbm-crite-v7-20260811`
Namespace: `VBMCRITEV7`

## Binding disposition

**CLOSED AT DEVELOPMENT SMOKE. NO FORMAL CAL/VAL. NO VOYNICH TARGET ACCESS.**

HF smoke job: `6a7b9f21f6d0f3ee953aa211`.

The exact-bigram-preserving Euler intervention is not an appropriate detector for the principal reusable homophonic-cipher controls. The reason is structural, not a thresholding failure.

The synthetic cipher family is a memoryless homophonic emission process driven by a first-order latent language Markov chain. Its observed likelihood is therefore overwhelmingly encoded in the observed surface bigram matrix. Exact preservation of that matrix preserves nearly all of the latent HMM likelihood as well.

Smoke median CRITE:

- BAV_GLOBAL: +0.000535
- GER_GLOBAL: +0.001736
- BAV_GLOBAL_SWAP: +0.006142
- BAV_FRESH: +0.003577
- GER_FRESH: +0.008417
- MARKOV1: +0.000014
- MARKOV2: +0.000187
- MARKOV3: +0.000404
- SLOT5: -0.001175

Thus a fresh-key German negative exceeded all three reusable-key positives except the swapped Bavarian control. The synthetic positive and negative distributions were already nonseparable in smoke.

More importantly, the per-fold latent surrogate excess for reusable-key controls was only about 0.001–0.008 nats/event: the exact pair-preserving surrogate deliberately retains the sufficient low-order information for this cipher family.

Formal CAL was therefore not launched. No threshold was frozen and no Voynich FIT/H1/C1 data were scored under v7.

## Successor

The next valid experiment must test whether the observed surface bigram matrix has a **reusable language-constrained latent factorization**, rather than randomizing away ordering while preserving that matrix.

The recommended successor is prospective held-out surface-pair completion:

1. mask a deterministic stratified subset of surface bigram cells in training;
2. fit Bavarian/German constrained factorisations without those cells;
3. fit strong non-language generic low-rank and empirical surface baselines to the same observed cells;
4. evaluate prediction of the masked pair cells on independent held-out folios;
5. qualify on same-key/fresh-key and stable non-language synthetic controls before any Voynich access.

This directly tests whether a compact reusable latent structure predicts unseen surface-pair relationships better than non-language surface structure.
