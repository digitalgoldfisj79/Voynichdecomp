# Voynich operational compositionality — v0.5 Track C result

**Status:** completed under the frozen v0.5 protocol.  
**Source:** 37,465 enriched token occurrences; complete-folio folds.  
**Interpretation boundary:** architecture only; no glyph meaning or domain assignment.

## Result

The strict predictive-codelength test failed. Adding frame or core identifiers to section/position models increased held-out codelength for next-prefix, next-suffix and next-length prediction. Cross-section transfer was negative for every tested outcome. The data do not support a compact universal transition code in which a sparse frame/core identity directly predicts the next packet.

A weaker but stable compositional effect survived. Token types sharing a control frame or a core have more similar local context distributions than frequency- and length-matched unrelated token types. The frame effect passed all five folds under both lossless P70 and the simpler no-suffix decomposition, and exceeded both order controls.

| Decomposition | Real frame effect | Within-line shuffle | Conditional resample |
|---|---:|---:|---:|
| P70 | 0.1007 | 0.0142 | 0.0700 |
| No suffix | 0.0730 | 0.0155 | 0.0544 |

Here the effect is the increase in context similarity over matched unrelated token pairs. Real-minus-control differences were positive in all five folds against both controls; exact one-sided sign-flip p = 0.03125 for each frame comparison.

The core effect also exceeded within-line shuffling under both segmentations. Against the stronger conditional-resampling control it was robust for the no-suffix analysis (mean difference 0.0131; p = 0.03125) but not for P70 (mean difference 0.0032; p = 0.15625).

## Adjudication

**Gate C passes narrowly on frame compositionality.**

The surviving statement is:

> Voynich token frames organise distributions of neighbouring forms beyond simple token frequency, length, section and line-position effects. This organisation is stable across two segmentation policies but is substantially weaker after conditioning on the manuscript marginals.

The result does not show that frames encode an operation, duration, register or mode. It shows a contextually coherent morphological frame. The failure of direct codelength transfer means the stronger operational-state interpretation remains unproved.

## Negative results retained

- No next-event predictive gain from sparse frame/core identity.
- No cross-section predictive transfer.
- P70 core compositionality does not beat the strongest conditional control at the exact five-fold level.
- The result is compatible with constrained morphology, abbreviation, cipher machinery or structured generation as well as operational notation.
