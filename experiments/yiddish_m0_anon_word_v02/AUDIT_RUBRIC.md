# Anonymous word v0.2 visual-audit rubric — frozen before Shmuel 1544 image access

This document resolves the phrase `substantive text line` in the already-frozen v0.2 protocol before any confirmation image is viewed. It does not change segmentation constants, `tau`, descriptors, page selection, or thresholds.

## Scope of the body-text instrument

The registered long cell is a sequence of **running body-text word segments**. Therefore the zero-missing-line visual gate concerns running body text, not every printed mark or every standalone paratextual line on the page.

### Counts as a substantive body-text line

A line is substantive for the omission gate when it forms part of the continuous prose/verse body and contains ordinary lexical word tokens whose omission would create a gap in that running body sequence.

### Does not count as a substantive body-text line

The following standalone material is paratext for this instrument and is not a missing substantive line when unretained:

- page/folio numbers and signatures;
- isolated stanza/section numerals;
- running headers/footers and catchwords;
- standalone titles or section labels/headings that are spatially separated from the running body;
- standalone colophonic/closing formulae spatially separated from the running body;
- ornaments, rules, printer marks, and isolated punctuation.

A line does **not** become paratext merely because its typography differs. If it continues a running sentence/verse or contains body words in the body block, it remains substantive.

## Boundary-error rule

The <=2 obvious word-boundary errors/page rule is evaluated only on retained substantive body lines. A boundary error is counted only where a retained box visibly joins two whitespace-separated body words or splits one visually continuous body word into two retained tokens. Omitted punctuation/ornament boxes are not boundary errors.

## Basel DEVELOPMENT classification under this rubric

The consumed Basel pages are used to fix the interpretation before confirmation:

- canvas 5: the unretained standalone `weiter ...` section label is paratext; all running body lines are retained.
- canvases 11, 17, 23, 29: all running body lines are retained.
- canvas 35: the short standalone heading continuation above the body and the separated closing `... amen` formula below it are paratext; all running body lines are retained.

This classification is frozen and must be applied identically to the fresh Shmuel audit. In particular, if Shmuel has an omitted line that continues its running body, it fails even if the line is short, indented, bold, or otherwise typographically unusual.
