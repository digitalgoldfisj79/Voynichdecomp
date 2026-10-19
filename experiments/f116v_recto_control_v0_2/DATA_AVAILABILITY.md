# Stage-2 acquisition result

Date checked: 2026-08-04

The public Google Drive archive was recursively enumerated with `gdown --folder --json`.

## Decisive finding

- Raw f116v TIFFs: **46**
- Raw f116r TIFFs: **0**
- Matching f116r multispectral cube: **absent from the public archive**

The archive contains raw MSI for ten selected folios: 1r, 8r, 17r, 26r, 47r, 70v1, 71r, 93r, 102v1 and 116v. This matches the published account of the 2014 capture set.

## Replacement control

The stage-2 executable uses Yale's high-resolution visible-light f116r IIIF image (`child_oid=1006276`) as a constrained show-through proxy and f115r/f115v (`child_oid=1006274`, `1006275`) as unrelated-folio controls.

This replacement is weaker than a spectrally matched recto cube. Therefore the programme prohibits a final `EVIDENCE_PRESENT` verdict. The strongest permitted outcome is a recto-independent candidate requiring the matching cube or new acquisition.
