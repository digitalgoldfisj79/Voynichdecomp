# f115r gallows-only boundary assay v0.2 — frozen protocol

Date: 2026-09-16

## Reason for v0.2

v0.1 word segmentation failed its own QC and its word/glyph morphology result is inadmissible. v0.2 removes word segmentation entirely.

## Target question

Does the upper morphology of EVA `k` (Davis v101 `h`) change at the proposed f115r Scribe-2 -> Scribe-3 boundary between physical lines 12 and 13, independently of the later ink/session change around lines 18/19?

## No hand-label tuning

No Davis hand labels are used during line detection, parameter calibration, gallows detection, gallows typing or feature extraction.

Lines 25–45 are the **development region**. They are used only to calibrate two nuisance parameters for upper-gallows detection:

1. fraction of each line strip treated as the upper/gallows band;
2. horizontal closing width used to join disconnected parts of a gallows top.

The objective is purely transcriptional: maximize the number of development lines on which the number of detected upper structures exactly equals the known count of EVA gallows characters `k/t/p/f`; tie-break by lower absolute count error and then the smaller closing width.

Lines 7–24 are the sealed target/control region and do not influence parameter choice.

## Detection and typing

For each physical text line:

- use the high-resolution Yale 2014 f115r scan (image ID 1006274);
- derive a dark-ink mask with the same fixed local threshold as v0.1;
- inspect only the calibrated upper band;
- horizontally join nearby upper-stroke pixels using the development-selected closing width;
- connected upper structures are ordered left-to-right;
- a line is **eligible only if detected structure count exactly equals the transcript's `k/t/p/f` count**;
- on eligible lines, detected structures inherit gallows type solely by left-to-right order from the transcription.

No expected word boundary or Davis writer label enters this mapping.

## Primary diagnostic

Only assigned EVA `k` structures contribute to the morphology assay. `t/p/f` are localization/QC controls and are not pooled into `k`.

Transparent upper-structure features are frozen as:

- normalized aspect ratio;
- ink occupancy;
- centroid slant (x drift with y);
- horizontal span in upper, middle and lower thirds;
- maximum horizontal-run ratio;
- upper-structure row dispersion;
- hole-area ratio where a closed loop is present;
- width normalized to physical line-strip height.

No learned embeddings, OCR, DINO, SAGHOG, HTR or cross-manuscript model is permitted.

## Independent unit

The physical line is the unit. Multiple `k` structures on one line are averaged before any contrast.

## Frozen contrasts

- Davis boundary: lines 7–12 versus 13–18.
- Ink/session boundary: lines 13–18 versus 19–24.
- Secondary: six-line matched windows around all other usable paragraph boundaries.

The omnibus statistic is RMS label-independent standardized mean difference across frozen features. Exact line-label permutation is used whenever feasible.

Minimum admissibility for the Davis contrast: at least four eligible physical lines containing `k` on each side, plus at least 70% exact gallows-count agreement across development lines with at least one gallows. If either fails, morphology verdict is `UNDERPOWERED_OR_QC_FAIL`.

## Ink colour

The separate line-level RGB/contrast assay is repeated with the same registered lines. It is a session/ink sanity check only, never evidence of writer identity.

## Interpretation

A significant 12/13 `k` morphology jump with no coincident colour jump, stronger than 18/19 and unusual among paragraph controls, supports Davis locally. A strong colour jump at 18/19 with morphology continuity at 12/13 supports a distinct ink/session boundary but not Davis's writer change. Failure of the 12/13 gallows contrast on an adequately qualified assay weighs against using f115r as independent validation of the two-hand claim, but does not by itself refute the full five-scribe model.
