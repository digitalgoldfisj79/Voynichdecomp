# VMS edge forensics physical-image addendum v0.1 — FROZEN BEFORE PAIR RANKS

Date: 2026-09-07
Parent: `vms_edge_forensics_orientation_20260907_v01`
Addendum ID: `vms_edge_physical_image_20260907_v01`

## Purpose
Test whether the surviving early-Herbal cross-quire text-state edges E1/E2 have unusual **upper-spill / edge-state compatibility** relative to matched early-Herbal bifolium pairs, without allowing candidate-specific image tuning.

This assay is a P2 stain/damage channel with a P1 edge-shape sensitivity channel. It cannot by itself establish chronological adjacency or direction.

## Eligible pool
- complete bifolia in current Q1–Q7 for which both leaves survive and all four page sides are available from Yale IIIF;
- the incomplete f12/f13 bifolium is excluded automatically;
- image features are computed for every eligible page before candidate labels are revealed.

## Acquisition
- Yale IIIF 2014 colour scans;
- width 900 px, as in the qualified S3 spill detector;
- exact downloaded JPEG SHA-256 stored;
- no image enhancement chosen by candidate.

## Frozen detector
Reuse `vms_spill_topology.py` S3 `edge_feature` unchanged:
- parchment-edge alignment;
- narrow top/bottom edge strip;
- dark-ink/high-saturation/large-gradient rejection;
- 6x32 low-frequency grid + 32-bin profile;
- identically processed bottom edge as matched scanner/parchment control.

No threshold in S3 may change in this assay.

## Physical coordinate handling
For each physical leaf:
- recto profile/grid kept in recto coordinates;
- verso is horizontally mirrored before aggregation;
- leaf vector = mean(recto, mirrored verso).

For each bifolium:
- retain its two leaf vectors as an unordered two-element set;
- do not use folio number/nesting position in distance calculation.

## Unit-to-unit distance
For bifolia A={a1,a2}, B={b1,b2}:
- compute the two possible one-to-one leaf assignments;
- distance = minimum of the mean pair distances across those two assignments;
- feature dimensions are robust-standardised over all eligible leaves before pair distances;
- this makes comparison invariant to unknown left/right orientation of a loose bifolium.

Metrics:
1. `TOP_PROFILE_DIST`: S3 grid+profile distance at upper edge.
2. `BOTTOM_PROFILE_DIST`: identical lower-edge distance.
3. `TOP_SPECIFIC_PROFILE = TOP_PROFILE_DIST - BOTTOM_PROFILE_DIST`; more negative = unusually upper-specific similarity.
4. `TOP_SCALAR_DIST`: distance in S3 mean/p90/area015/area025 upper metrics.
5. `BOTTOM_SCALAR_DIST`: matched lower metrics.
6. `TOP_SPECIFIC_SCALAR = TOP_SCALAR_DIST - BOTTOM_SCALAR_DIST`.
7. `EDGE_SHAPE_DIST`: top/bottom parchment edge-y mean/SD only; compatibility/sensitivity, never sufficient for promotion.

## Blinding
- compute and persist all eligible bifolium-pair scores with opaque hashed IDs first;
- hash/salt mapping is written to a separate reveal file only after pair-score file is closed;
- candidates E1/E2 are ranked only in reveal stage.

## Matched null
For each candidate use alternatives whose two units match the same IVTFF Currier + hand + broad illustration class pair where >=20 endpoint-disjoint alternatives exist; then Currier+hand; then Currier+illustration; then all early-Herbal complete bifolia.

Candidate endpoints are excluded from the null.

## Positive criterion P2
`P2_QUANT_SIGNAL` requires ALL:
- TOP_SPECIFIC_PROFILE is at least 2 matched-null SD more negative than null mean;
- exact/empirical lower-tail p<=.01;
- TOP_SPECIFIC_SCALAR has the same direction (need not independently reach 2 SD);
- the profile result survives leave-one-page-side-out recomputation for each of the eight page sides contributing to the two bifolia: every deletion must retain the same effect direction, and at least 6/8 deletions must remain >=2 SD;
- bottom-edge raw similarity is not itself >=2 SD more extreme in the same direction.

If |effect|/null SD <2: `the metric does not resolve this`.

## Known detector calibration
The existing f32v/f33r discontinuity recovery remains a detector-sensitivity check only. It must not be counted as candidate support.

## Interpretation
- A P2 signal means the bifolia share unusually similar upper-edge damage/stain state relative to the bottom control.
- It does NOT establish they were directly adjacent, because a spill can affect multiple nearby sheets similarly.
- Stage-F promotion still requires the parent protocol's independent documentary/physical corroboration and no contradiction.

## Stop rule
If neither E1 nor E2 passes P2, do not retune the detector or broaden image regions in v0.1.
