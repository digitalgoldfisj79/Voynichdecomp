# Voynich recoverability-first open-set programme — final v0.5 report

**Date:** 2026-07-25  
**Branch:** `experiment/voynich-recoverability-open-set-v0.5-20260725`  
**Formal programme verdict:** **CALIBRATION_FAILURE**  
**Substantive Voynich status:** **structure without demonstrated recoverability**

## 1. Objective

The programme replaced the question “Which notation family does Voynich resemble?” with the stronger question:

> Does Voynich contain recoverable, transferable operational variables characteristic of human technical notation, under identity-neutral representations and explicit open-set abstention?

The protocol was frozen before the formal external and Voynich outputs were generated. The v0.4 `ABSTAIN_OOD` verdict remains unchanged.

## 2. External calibration

### Representation-neutral recognition passed

Known historical notation remained strongly recognisable after replacing literal signs with frequency ranks, equality patterns, run patterns, event ranks and length-only representations. Grouped ensemble ROC AUC ranged from 0.9896 to 0.9997; all no-length variants exceeded 0.98.

This shows that the historical classifier is detecting substantial structural differences rather than merely memorising alphabets or token lengths.

### Known-field recovery did not meet the frozen gates

Ammerbach pitch/rest and duration/special channels could not be aligned across books at the required reliability. Zero-offset top-1 recovery ranged from 0.4475 to 0.5492, below the 0.70 threshold.

Square-neume boundaries were recoverable above threshold (F1 0.7546 versus baseline 0.5966). Aquitanian boundaries were not: F1 0.6685 versus baseline 0.6583. Because both families were required, the neume gate failed.

Therefore:

- Gate A: **pass**;
- Gate B1: **fail**;
- Gate B2: **fail**;
- formal downstream open-set architecture comparison: **inadmissible**.

## 3. Voynich compositionality

The strict operational prediction failed. Sparse frame or core identities did not improve held-out prediction of the next prefix, suffix or token length, and cross-section transfer was negative for every tested outcome.

A weaker compositional property survived. Tokens sharing a control frame had more similar neighbouring-token distributions than frequency- and length-matched unrelated tokens. This held in all five folio folds under both lossless P70 and the simpler no-suffix segmentation, and exceeded both within-line shuffle and conditional-resampling controls.

| Decomposition | Real frame effect | Line shuffle | Conditional resample |
|---|---:|---:|---:|
| P70 | 0.1007 | 0.0142 | 0.0700 |
| No suffix | 0.0730 | 0.0155 | 0.0544 |

The exact one-sided five-fold sign-flip probability was 0.03125 for each frame-versus-control comparison.

This supports a contextually coherent morphological frame. It does not establish that the frame encodes an operation, duration, register, target or mode.

## 4. Adjudication

The formal v0.5 verdict is `CALIBRATION_FAILURE`, because the programme could recognise historical notation but could not recover its known operational structure reliably enough to authorise the final transfer test.

The scientifically useful Voynich result is narrower:

> Voynichese contains stable frame-conditioned contextual structure, but the present programme did not recover a compact, cross-section operational code.

Accordingly:

- literal musical notation: **not supported**;
- membership in tested neumatic or organ-tablature families: **still rejected at v0.4 resolution**;
- broad operational notation: **not disproved, but not recovered**;
- four-field P70 operational packet: **not established**;
- contextually coherent affix/frame morphology: **supported**;
- specific procedural or synthetic family assignment: **not permitted**.

## 5. Consequence for the Cryptologia framework

The programme validates the paper's evidential hierarchy in practice:

1. surface resemblance and high classification accuracy are insufficient;
2. identity-neutral recognisability can survive while known-field recoverability fails;
3. when recoverability calibration fails, the unknown target must not be assigned a mechanism;
4. the correct output is a documented failure or abstention, not a nearest-class story.

This is a useful negative empirical companion to the methodological paper.

## 6. Remaining meaningful route

The next phase should not tune another Voynich-only classifier. It requires better ground-truth technical corpora:

- fifteenth-century German organ and lute tablature with aligned surface and event streams;
- larger Aquitanian corpora with reliable graphical event boundaries;
- real medieval recipes, computus, pharmaceutical tables, dosage notation and administrative shorthand represented as both surface signs and operational fields.

Only then can the recovery gates be recalibrated and the sealed open-set test repeated. Until those data exist, the encoded-analysis programme has reached a genuine evidential stopping point rather than a computational one.

## 7. Reproducibility

- Frozen protocol: `FROZEN_PROTOCOL_v0_5.md`
- External runner reconstructed from four independently compressed chunks; verified source SHA-256: `3f15c8e7b344a0ca0069b8cb4f0717aa2fc26ae39376f04c0c8c8ec920ec8648`
- External execution: Hugging Face Job `6a6458dcdb23d7a7ec1cbab8`
- External result SHA-256: `83299da49877ce6183b2d895dc3533f82083da0c815a33121c9a907004ca9ad1`
- Track C full-result SHA-256: `c61ad8f66cbad87cdd3791c6120592c0473fe65c29d6df2be83203d9943db150`
