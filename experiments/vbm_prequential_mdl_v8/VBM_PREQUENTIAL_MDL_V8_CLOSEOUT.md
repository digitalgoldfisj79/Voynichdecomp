# VBM v8 — prequential / MDL closeout

Date: 2026-08-11
Branch: `experiment/vbm-prequential-mdl-v8-20260811`
Namespace: `VBMPREQMDLV8`

## Binding result

**CLOSED AT FORMAL CALIBRATION: PREQ/MDL NONSEPARABLE FROM STABLE MARKOV1 CONTROL.**

No VAL and no Voynich FIT were run.

HF smoke job: `6a7ba265f6d0f3ee953aa236`.
HF formal CAL job: `6a7ba2aaf6d0f3ee953aa238`.

Primary statistic: `PREQ_ADV = (NLL_surface - NLL_latent) / coded_events`, where positive values favour the constrained Bavarian/German latent model.

Formal CAL used three fresh replicates per family.

Reusable-key positives:
- BAV_GLOBAL: +0.233908, +0.249627, +0.191645
- GER_GLOBAL: +0.365565, +0.355588, +0.368971
- BAV_GLOBAL_SWAP: +0.170269, +0.214705, +0.182154

Fresh-key controls:
- BAV_FRESH: -0.021712, -0.024472, -0.004353
- GER_FRESH: +0.028613, +0.008111, +0.007838

Stable non-language controls:
- MARKOV1: +0.209344, +0.233100, +0.200977
- MARKOV2: -0.040094, -0.013378, -0.045812
- MARKOV3: -0.076993, -0.068889, -0.075370
- SLOT5: -0.023899, -0.021997, -0.026220

Calibration extrema:
- weakest reusable-key positive: +0.1702688567
- strongest negative/adversarial result: +0.2330995206 (MARKOV1)

No separating threshold exists. Formal verdict: `PREQ_nonseparable` — FAIL.

All 9/9 reusable-key positive replicates had positive PREQ_ADV, and the correct Bavarian/German source language was selected in 5/6 global-key CAL replicates. These do not rescue the instrument because MARKOV1 is a binding adversary and overlaps the positive range.

## Interpretation

The prequential compression route does not resolve the existing identifiability problem. A stable surface Markov process generated from the same induced law can receive essentially the same compression benefit from the constrained latent factorisation as genuine reusable-key ciphertext. Thus predictive compression advantage is not specific enough to distinguish the two model classes here.

This is not evidence that Voynich is or is not encrypted. It is a negative result for the current VBM + Bavarian/German first-order homophonic-transducer programme.

## Disposition

- CAL failed.
- VAL not run.
- Voynich FIT not run.
- H1/C1 not reused.
- no decoding analysis performed.
- no successor experiment proposed or run.

The current branch is closed.