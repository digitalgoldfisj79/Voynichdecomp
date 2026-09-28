# XD1-CHARM v1.1 pre-outcome addendum

Frozen on 2026-09-28 after source-only inspection and **before any charm recurrence statistic was computed**.

This addendum records two source-only corrections to the original protocol.

1. Cambridge CUDL encodes diplomatic physical lines in the rendered transcription as source-derived `<br>` objects carrying transcription-line IDs and polygon geometry (`data-points`). The first preflight searched only for TEI `<lb>`/`<line>` and therefore produced a false-negative lineation check. This was corrected before opening any recurrence outcome.
2. Of the four Cambridge University Library manuscripts independently listed in Taylor (2025), Add. 9308 exposes diplomatic physical lines in CUDL (183 pages). Dd.vi.29, Dd.5.76 and Dd.iv.44 expose zero CUDL diplomatic pages and are excluded from the primary metric rather than reconstructed from images/OCR. Dd.5.53 and Dd.3.52 likewise do not expose diplomatic pages through this route.

The machine-readable primary manifest is `research/XD1_CHARM_MANIFEST_20260928.json`.

## Primary population

Fourteen independently source-defined Add. 9308 charm units are frozen. Ten form a stricter sensitivity subset because the manuscript explicitly labels them “A charm” (or the text explicitly calls itself a charm); four additional units are catalogued or independently treated as charms in the Curious Cures / specialist literature.

**Whole-line rule:** include only manuscript physical lines wholly within a charm. If a single physical line contains both the end of an adjacent recipe and the charm rubric/body, or both the end of a charm and the next recipe, exclude that line completely. Never split a manuscript line.

This deliberately sacrifices some charm text at boundaries in exchange for an unambiguous physical-line statistic.

## Parser freeze

Before scoring, the CUDL parser must:
- isolate/remove `<script>` and `<style>` content before line extraction;
- require a source line ID and/or polygon geometry;
- preserve the diplomatic text exactly except for the already-frozen XD1 tokenisation;
- assert that known page-final lines contain no CUDL JavaScript.

Any parser correction after recurrence output is opened requires a new protocol version.

## Secondary within-manuscript background

For descriptive context only, score the Add. 9308 main-compilation physical lines (ff.1v-89r) after excluding every primary charm line and every frozen mixed-boundary line. Call this **medical background**, not “pure recipes”: unidentified charms may remain. It is not part of the downgrade gate.

All original XD1-CHARM downgrade thresholds remain unchanged.
