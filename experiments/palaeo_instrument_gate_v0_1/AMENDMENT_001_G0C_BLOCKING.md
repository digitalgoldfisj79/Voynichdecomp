# Amendment 001 — G0C manuscript-level blocking correction

**Date:** 2026-08-08

**Status:** frozen after G0A was observed but before any G0C result was observed.

## Trigger

Code review during execution identified that `run_gate0.py` permutes source labels at crop level. Digitisation source is a manuscript-level property, so crop-level permutation would pseudo-replicate the five-manuscript G0C stress panel and produce an invalid significance test.

This correction is based on the design structure, not on any G0C metric.

## Binding correction

1. The G0C source-leakage **effect metrics** from the initial job (directional AUC, balanced accuracy, pooled AUC), if emitted, remain descriptive diagnostics.
2. The crop-level permutation p-value from that job is **scientifically void** and may not be used in a gate decision.
3. Any inferential permutation must assign source at the manuscript block level, with all crops from a manuscript moving together.
4. The current crossed panel contains only two corridor manuscripts and three Bavarian-control manuscripts. Within-domain manuscript-label permutation therefore has too few unique assignments to support a p <= 0.05 criterion.
5. Consequently the current Stage-5-derived corpus has no possible `G0C PASS` and no formal inferential `G0C FAIL`. Its maximum status is `DIAGNOSTIC_LEAKAGE` or `INDETERMINATE`.

## Interpretation rule

- If both cross-domain directional AUCs are >=0.65 and their mean is >=0.70 for any representation, record **DIAGNOSTIC_LEAKAGE**: residual source signal transfers across geography and the representation is unsafe pending a larger fixed-glyph/source-crossed calibration.
- Otherwise record **INDETERMINATE**.

A future formal G0C must use a larger manuscript-blocked, fixed-glyph or homologous-form panel with enough independent manuscripts for a meaningful equivalence test around chance.
