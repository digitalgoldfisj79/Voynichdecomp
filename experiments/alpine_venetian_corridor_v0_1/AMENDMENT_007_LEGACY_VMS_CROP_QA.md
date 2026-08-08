# Amendment 007 — Mechanical QA for legacy VMS plant/root references

Date: 2026-08-08
Programme: Alpine–Venetian Corridor Illustration Programme v0.1
Run: `corridor_v01_20260808_run01`

## Reason

The existing Manucomp/VMS catalogue contains reviewed plant and root boxes created before this corridor experiment. Audit before any corridor-to-VMS similarity was computed showed that two reviewed root boxes are pathological sliver crops. Passing such crops to DINOv3 would be an input-quality error rather than evidence.

An initial draft of this amendment incorrectly described the legacy frame as a universal 1000 x 1300 canvas. A database audit, still before any VMS/comparandum similarity, showed that the catalogue normalises each folio to width 1000 while retaining its aspect ratio, so the corresponding normalised height varies by folio. This file records the corrected rule rather than preserving that mistaken fixed-height assumption.

## Frozen mechanical rule

For legacy `cat_objects` rows used as VMS image references:

- `reviewed = true` remains necessary but is not sufficient;
- normalised crop width must be >= 50 pixels;
- normalised crop height must be >= 50 pixels;
- x/y must be non-negative;
- x + width must be <= 1000;
- y + height must be <= `ceil(cat_folios.height * 1000 / cat_folios.width)`, the folio-specific normalised height;
- no failed crop may be manually enlarged, merged, redrawn or rescued after corridor results are visible.

The rule is class-independent and applied before any comparison score is computed.

## Consequence at freeze

Database recomputation gives:

- plant: 8 reviewed, 8 pass, 0 fail;
- root: 17 reviewed, 15 pass, 2 fail.

The two excluded reviewed roots are:

- f22v, seq 44: x=560, y=800, w=400, h=40;
- f65r, seq 117: x=4, y=724, w=921, h=1.

No other reviewed plant/root reference is excluded by this rule.

## Coordinate conversion

Legacy boxes use a width-normalised frame. Let `Hn = cat_folios.height * 1000 / cat_folios.width`. Source-image crop fractions are therefore:

- x0 = `x / 1000`;
- x1 = `(x + w) / 1000`;
- y0 = `y / Hn`;
- y1 = `(y + h) / Hn`.

The corresponding full-resolution source is resolved from `cat_folios.iiif_service`. This conversion preserves the original folio aspect ratio rather than assuming a fixed height.

## Outcome firewall

At this corrected amendment:

- `vms_similarity_computed = false`;
- no corridor/control crop has been compared with any VMS crop;
- the correction was prompted solely by metadata geometry, not by an outcome;
- this rule may only exclude malformed VMS reference crops and cannot change a corridor/control candidate, match, object class, date, geography, threshold or positive-result criterion.
