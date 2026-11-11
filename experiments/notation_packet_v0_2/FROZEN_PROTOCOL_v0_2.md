# Frozen protocol reconstruction — Voynich notation falsification v0.2

This file records the design implemented before the outputs in the v0.2 scripts were inspected.

## Primary estimand
Held-out codelength difference, in bits per token, between a soft HMM and a parameter-matched IID latent mixture. Both models use the same latent state count and the same prefix/gallows/suffix emissions. Core probabilities remain section-conditioned and outside the latent state.

## Units and splitting
Complete folios are held out. Five deterministic section-stratified folio splits use seeds 101, 202, 303, 404 and 505. K=8 and K=12 are both reported.

## Primary gate
The HMM-vs-IID codelength gain must be positive for every split at both state counts.

## Segmentation gate
P70's within-segmentation direct packet gain must exceed the best matched alternative segmentation by at least 0.10 bits/token. This gate is retained even if it fails. Absolute codelength is secondary because a full comparison should also charge segmentation/model description length.

## Alternative segmentations
P70; conventional ch/sh-as-gallows; no-ch/sh-prefix; shifted c+h; flat no-gallows; fixed 1/2 cut; inventory-matched random split.

## Calibration
Planted stateful and IID packet controls use arbitrary surface-symbol permutations, omitted control fields, complete-folio holdouts and true K=6. Report HMM-vs-IID, aligned state accuracy and adjusted Rand index.

## Transfer and operational diagnostics
Leave-one-section-out packet transfer; leave-one-Davis-hand-out packet transfer; within-section cross-hand transfer; previous-suffix to next-prefix held-out gain; prefix line-start odds; f115r blind change-point scan.

## Interpretive boundary
No semantic assignment to a glyph is permitted. Positive structural fit is not a decipherment or identification of musical notation.
