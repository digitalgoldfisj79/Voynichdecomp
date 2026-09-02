# Fractionation Signature Gate v0.1

Status: preregistered before any Voynich evaluation.

## Question

Can surface statistics identify a bounded two-coordinate / two-fraction production process strongly enough to distinguish it from matched non-fractionated and observationally equivalent polygraphic controls?

This is an attribution gate, not a plaintext-recovery exercise. The Voynich Manuscript remains sealed unless the synthetic gate passes.

## Positive family

Each plaintext symbol receives a keyed pair of latent coordinates `(r,c)`. Coordinate values are rendered through fresh row/column symbol inventories. For block width `b`, the output is regrouped as `r1...rb c1...cb`. Keys are independently randomized per sample.

Two variants are crossed:

1. clean: disjoint coordinate-role inventories;
2. noisy: bounded homophony, bounded row/column symbol overlap, and low-rate null insertion.

Train block widths are `{2,3,4,6,8}`. Locked-test widths are unseen `{5,7,9,10,11,12}`.

## Controls

### Easy bigraphic control

Each plaintext symbol receives an arbitrary ordered pair of visible symbols, emitted adjacently without coordinate-role regrouping. This matches the 2:1 expansion but not the two-stream block geometry.

### Observational-twin control

An arbitrary keyed bigraphic code is rendered with the same two-stream regrouping law, key randomisation, symbol-role inventories, and noise model as the positive family.

This control is intentionally severe. A keyed Polybius square with arbitrary cell assignment and an arbitrary injective bigraphic code over the same row/column product can induce the same observable distribution. If the detector cannot distinguish them, coordinate/Polybius semantics are not identifiable from these surface statistics.

## Corpus and leakage control

Reuse the six frozen v0.5 Universal Dependencies corpora and their pinned hashes: English, German, Finnish, Turkish, Hebrew, Arabic. Training samples come from the existing train split; locked evaluation samples come from the existing test split. No Voynich text is used for feature design, training, threshold selection, or debugging.

Sequence length is fixed at 384 source characters before removal of spaces and encoding. Replicates default to 32 per language per split.

## Features

All features are invariant to arbitrary renaming of visible symbols:

- visible vocabulary size and entropy;
- normalized mutual information at lags 1–12;
- candidate two-role separation measured by Jensen-Shannon divergence for block widths 1–12;
- summary statistics of the block-width scan and the maximizing candidate width.

A random forest is trained only on synthetic train samples.

## Primary decision rule

For each binary discrimination, report locked-test ROC AUC, effect above chance (`AUC - 0.5`), permutation-null SD, and `effect / null SD` in the same result record.

The easy control must satisfy both:

- AUC >= 0.80;
- effect/null-SD >= 2.

The observational twin must satisfy both:

- AUC >= 0.65;
- effect/null-SD >= 2.

Only if both pass is the decision `GO_TO_VOYNICH`.

If the easy control fails: `STOP_DETECTOR_WEAK`.

If the easy control passes but the observational twin fails: `STOP_NON_IDENTIFIABLE`.

No threshold may be changed after the locked synthetic run.

## Secondary parameter-recovery check

On positive locked-test samples, scan candidate widths 1–12 and estimate the block width by the maximum role-separation score. Report exact-width and within-one recovery for clean and noisy variants. This is diagnostic only and cannot override the primary gate.

## Interpretation constraints

A positive easy-control result establishes detectability of two-stream regrouping, not Polybius semantics.

Failure against the observational twin means the metric does not resolve coordinate fractionation from a generic polygraphic structural twin. In that event no Voynich run is permitted under v0.1.

A future broader test would require an independently motivated observable that differs between the historical hypotheses; adding ad hoc repair mechanisms after seeing Voynich is prohibited.
