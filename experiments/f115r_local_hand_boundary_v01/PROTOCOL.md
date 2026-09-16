# f115r local hand-boundary assay v0.1 — frozen protocol

Date: 2026-09-16

## Question

Does the palaeographic morphology on f115r change at Lisa Fagin Davis's proposed Scribe-2 -> Scribe-3 boundary (physical text lines 12/13), independently of the later visible ink-colour change near lines 18/19?

This is a same-page assay only. It makes no cross-manuscript provenance claim.

## Inputs

- Yale 2014 scan mirror image ID 1006274 (f115r).
- Stolfi/Zandbergen f115r line transcription from voynich.nu.
- Davis diagnostic forms only:
  1. v101 `h` = EVA `k` (single-loop gallows): vertical/slant, crossbar/upper-loop geometry, compactness, final-tick geometry where measurable.
  2. word-final `m/n`: length and height of the terminal return stroke where measurable.

No learned image embedding, no DINO/SAGHOG representation, no OCR, and no cross-manuscript classifier is permitted.

## Blind extraction

Image processing receives only physical line number and transcription token. It does not receive Davis hand labels while locating lines, words, or glyph candidates or while calculating morphology features.

Word segmentation uses the known number of transcript tokens on each line only to choose inter-word whitespace boundaries. For EVA `k`, only tokens containing exactly one `k` and no other tall gallows (`t`, `p`, `f`) are eligible for automatic isolation. This is intentionally conservative.

All candidate crops and segmentation overlays are persisted for QC.

## Frozen contrasts

Primary hand-boundary contrast: lines 7–12 versus 13–18. This brackets the proposed 12/13 hand change while remaining on the dark-ink side of the reported later ink transition.

Primary session/ink contrast: lines 13–18 versus 19–24.

Secondary scan: identical six-line windows around every paragraph boundary on the page, where enough eligible glyphs exist. The Davis boundary is compared with this within-page boundary distribution.

## Unit of inference

The independent unit is the physical text line, not the glyph. Multiple eligible glyphs on one line are first averaged to a line-level feature vector. No glyph-level pseudoreplication is allowed.

For the primary six-line contrast, the omnibus shape statistic is the root-mean-square of label-independent standardized mean differences across frozen transparent morphology features. Exact line-label permutation is used wherever the eligible-line count permits it. Individual feature effects are reported descriptively.

Word-final m/n is expected to be sparse and is secondary/descriptive unless at least four eligible physical lines occur on each side of a contrast.

## Ink-colour sanity check

Per-line ink colour/darkness is measured separately from morphology. A colour jump at 18/19 but not 12/13 would support a pen/ink-session change distinct from Davis's proposed hand boundary. Colour is never interpreted as writer identity.

## Decision categories

- `DAVIS_BOUNDARY_SUPPORTED`: reproducible morphology discontinuity at 12/13 stronger than the 18/19 session contrast and unusual among matched paragraph boundaries, without a coincident colour jump.
- `INK_SESSION_ONLY`: stronger morphology/colour discontinuity at 18/19, with no comparable 12/13 morphology jump.
- `AMBIGUOUS_OR_UNDERPOWERED`: segmentation/QC failure, sparse diagnostics, or neither contrast clearly dominates.
- `F115R_BOUNDARY_CONTRADICTED`: adequately powered diagnostics are continuous across 12/13 and the proposed boundary is not unusual relative to within-page variation.

No category generalizes automatically from f115r to all five Davis hands.
