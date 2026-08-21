# SVT v0.4.3 — Length-comparable joint semi-Markov gate

Status: **FROZEN BEFORE BINDING EXECUTION**

## Motivation
SVT v0.4.2 failed its binding gate. All eight fresh trials under-segmented. Code audit identified a sufficient scoring defect: candidate segmentations with fewer inferred plaintext units accumulated fewer negative raw language-model log-probability terms, making scores across different unit counts non-comparable.

v0.4.3 changes only the cross-segmentation/cross-structure scoring objective. The qualified v0.3.4 state/key optimiser, v0.4 surface model, candidate structures, beams, shortlist size, alternation count, primitive-period canonicalisation and binding thresholds are otherwise retained.

## Length-comparable language evidence
For decoded plaintext sequence x_1..x_n, define contextual language evidence

    G = sum_{i>=3} [ log P(x_i | x_{i-2}, x_{i-1}) - log P(x_i) ].

This is a log likelihood ratio between the trained trigram model and the same-length unigram baseline. The first two symbols contribute zero language evidence. It therefore measures contextual information rather than absolute sequence probability and removes the automatic advantage obtained merely by proposing fewer plaintext units.

The surface segmentation term remains the fitted semi-Markov log likelihood from v0.4, including the frozen unit-length prior (0.30, 0.45, 0.25) for lengths 1,2,3.

The joint selection score is

    J = surface_log_likelihood + G - BIC_key_schedule_complexity.

No fitted weight is introduced: all three terms are log-evidence / log-likelihood scale, and the surface term is used with coefficient 1.0.

Within any fixed segmentation, the existing v0.3.4 factorised key optimiser continues to use its frozen raw language score because all competing keys then decode the same number of units. Only comparisons that can change segmentation length use J.

## Binding data
- ISO: de
- split: dev
- plaintext length: 1536
- modes: periodic and line_reset
- replicates per mode: 4
- fresh binding namespace: offsets 27000..27003
- total trials: 8
- decoder input: unsegmented verbose surface, observed surface line starts, pinned language model, deterministic trial seed
- hidden from decoder: true unit boundaries, plaintext, mode, period, key
- Voynich: SEALED

## Search
- structures: periodic/line_reset x periods 2..12
- cheap beam: 160
- full beam: 320
- cheap key starts: 1
- full key starts: 12
- full alternations: 3
- shortlist K: 6
- primitive-period canonicalisation: proper divisors of selected period, same objective

## Frozen PASS gate
All conditions must hold:
- exact canonical (mode, primitive period): 8/8
- mean sequence recovery >= 0.90
- median sequence recovery >= 0.90
- minimum sequence recovery >= 0.85
- mean boundary F1 >= 0.90
- minimum boundary F1 >= 0.85
- mean absolute unit-count relative error <= 0.05

Any failure remains a failure. No post-hoc shortlist widening, threshold relaxation, truth-informed count penalty or namespace reuse is permitted.
