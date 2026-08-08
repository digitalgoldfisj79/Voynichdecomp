# Amendment 007 — Mechanical QA for legacy VMS plant/root references

Date: 2026-08-08
Programme: Alpine–Venetian Corridor Illustration Programme v0.1
Run: `corridor_v01_20260808_run01`

## Reason

The existing Manucomp/VMS catalogue contains reviewed plant and root boxes created before this corridor experiment. Audit before any corridor-to-VMS similarity was computed showed that several reviewed root boxes are only a few pixels high in the standardised 1000 x 1300 coordinate system. Passing these sliver crops to DINOv3 would be an input-quality error rather than evidence.

## Frozen mechanical rule

For legacy `cat_objects` rows used as VMS image references:

- `reviewed = true` remains necessary but is not sufficient;
- standardised crop width must be >= 50 pixels;
- standardised crop height must be >= 50 pixels;
- the box must lie within the 1000 x 1300 catalogue frame;
- no failed crop may be manually enlarged, merged, redrawn or rescued after corridor results are visible.

The rule is class-independent and applied before any comparison score is computed.

## Consequence at freeze

All eight reviewed plant boxes satisfy the rule.

The following reviewed root boxes fail solely on minimum height and are excluded mechanically:

- f22v, seq 44: h=40;
- f65r, seq 117: five micro/sliver boxes with h=25, 18, 31, 4 and 1.

Approximately eleven reviewed root boxes remain eligible. Exact counts are recomputed from the database during the scoring build rather than hard-coded.

## Coordinate conversion

Legacy boxes are in a standardised 1000 x 1300 frame. Source-image crop fractions are therefore `x/1000`, `w/1000`, `y/1300`, `h/1300`. The corresponding Yale IIIF image is resolved from `cat_herbal_folios.iiif_service` / `image_url`.

## Outcome firewall

At this amendment:

- `vms_similarity_computed = false`;
- no corridor/control crop has been compared with any VMS crop;
- this rule may only exclude malformed VMS reference crops and cannot change a corridor/control candidate, match, object class, date, geography, threshold or positive-result criterion.
