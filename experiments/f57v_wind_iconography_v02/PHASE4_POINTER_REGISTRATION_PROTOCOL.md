# PHASE 4 PREREGISTRATION — pointer/annulus registration

Date: 2026-08-09
Status: FROZEN BEFORE MEASUREMENT

## Motivation

Deep research after Phase 3 raised a representational alternative not captured by the earlier wind-vs-nonwind comparison: the four f57v humans may be **diagrammatic operators/indexers** rather than four equivalent personifications. Medieval `Image du monde` manuscripts independently attest humans placed around circles to encode movement and relative position, while tetradic cosmological traditions correlate multiple quaternities through circular/orthogonal structures.

The decisive test is therefore whether the visible f57v gestures are geometrically registered to the annular information architecture.

## Frozen target observations

Page-position labels only:

- N: faces outward; 2 secure gestures approximately E and W.
- E: faces inward; 1 secure gesture approximately W.
- S: faces outward; 2 secure gestures approximately E and W.
- W: faces inward; 1 secure gesture approximately S.

Exactly **six secure gesture vectors** enter the primary test. No hypothetical seventh/eighth gesture may be added after looking at ring intersections.

## Competing models

### O — operator/indexer
Gestures encode how to read, select, compare, traverse or transform annular material. Prediction: extended gesture vectors should register with independently detected structural anchors more strongly than chance.

### P — personification
Figures primarily embody four entities/categories. Gesture endpoints need not register reproducibly with textual/ring structure. Prediction: registration statistic near rotation/permutation null.

### Q — paired-opposition correlation
The diagram chiefly encodes two orthogonal pairs (e.g. hot/cold and wet/dry, or corresponding elements/seasons/humours). Prediction: axis-level structure may be strong even if individual gesture-to-token registration is weak.

### W — wind/directional submodel
A directional/anemological interpretation predicts additional 4/8/12 periodicity or directional residue. It is tested only as a secondary structural hypothesis; no wind-specific semantic reading is allowed from geometry alone.

## Data freeze

Primary target image must be a fixed high-resolution f57v source. Record SHA-256 of every image/crop used.

No contrast enhancement or crop is chosen after viewing intersection scores. If multiple image variants are available, predeclare all variants and report them separately.

## Stage A — diagram geometry, independent of text

1. Estimate diagram centre from the circular annuli using robust circle/ellipse fitting.
2. Identify each of the six secure gesture origins and endpoints from frozen image crops.
3. Represent each gesture as a ray from proximal arm/hand direction through the fingertip/endpoint.
4. Record angular uncertainty as an interval, not a single exact bearing where finger direction is ambiguous.

Primary vector coding must be performed without access to text-anchor labels.

## Stage B — structural anchor extraction, blind to gestures

Independently derive candidate ring landmarks from the full diagram while masking the human figures and gesture regions:

- annulus boundaries;
- radial separators/gaps;
- unusually large inter-token gaps;
- starts/ends of coherent text runs;
- repeated glyph/token anchors where transliteration coordinates can be mapped reliably;
- local changes of writing orientation or baseline;
- non-text graphical marks intersecting rings.

The anchor detector/coder must not see gesture vectors or their projected intersections.

## Primary statistic

For each gesture ray and each annulus it intersects, calculate angular/arc distance to the nearest independently detected structural anchor on that annulus.

Define normalized nearest-anchor distance `d_ij` relative to local anchor spacing/circumference.

Primary score:

`S = mean(exp(-d_ij / tau))`

with `tau` frozen before target evaluation from synthetic/positive-control diagrams, or use mean normalized nearest-anchor distance if no positive-control calibration is available.

The exact statistic implementation and `tau` must be frozen before unmasking target gesture bearings against anchors.

## Nulls

At minimum:

1. **global rotation null** — rotate the complete six-vector configuration around the fitted centre through all admissible angles while preserving vector origins relative to the centre where geometrically meaningful;
2. **within-position angle null** — randomize each gesture bearing within the visually possible sector while preserving origin;
3. **gesture-label permutation** — permute observed vector directions among the four figure positions;
4. **anchor rotation null** — rotate each annulus's anchor set relative to the frozen vectors, preserving within-ring spacing.

Use exact/enumerated rotations where resolution permits, otherwise >=10,000 Monte Carlo draws with fixed RNG seed.

## Primary success rule

Operator registration is supported only if:

- observed primary score is in the top 5% of the global-rotation null (one-sided p <= .05), AND
- effect survives at least two of the other three null families, AND
- result is not driven by a single figure or single annulus (leave-one-figure-out and leave-one-annulus-out directionally positive).

No post-hoc selection of rings, vectors or anchor types is allowed for the primary result.

## Secondary structural tests

### Periodicity
Test angular autocorrelation / spectral support for 4-, 8- and 12-fold segmentation of annular anchors/text blocks. Report all tested harmonics; do not choose only the best.

### Opposed-axis test
Compare structural descriptors sampled in sectors associated with N/S versus E/W figures. Candidate descriptors:
- text density;
- token/glyph distribution;
- paragraph/run boundaries;
- ring interruptions;
- local radial marks.

The N/S vs E/W contrast is primary within this secondary family because it was frozen before this phase.

### 4×3 test
If 12 significant anchor sectors exist independently, test whether they cluster as four triplets better than a uniform 12-sector model. This discriminates zodiacal-triplicity-like or principal/subordinate architectures from simple 12-fold wheels.

## Positive controls

Before target inference, the geometric pipeline must recover intentional registration on at least two historical or synthetic diagrams with known pointer/ray-to-label relationships. If it cannot, return **NO TEST**.

A negative control should include a circular manuscript diagram with decorative/personification gestures not intended as textual pointers, if one can be sourced independently.

## Interpretive decision matrix

- Registration PASS + opposed-axis structure: operator/indexer model rises strongly.
- Registration FAIL + strong opposed-axis ring differences: paired-opposition/correlation model rises.
- Registration FAIL + 4/8/12 periodicity with directional/wind residue: personification/wind or astrological sector model remains viable.
- Registration FAIL + no structured periodicity: figures are more likely iconographic/personificatory than mechanical pointers, or current image/transcription resolution is inadequate.

## Prohibited inference

This experiment cannot by itself identify the semantic values of the four positions. Even a strong pointer result would establish a *diagrammatic role*, not whether the content is winds, elements, humours, ages, directions, astrology, or another tetradic system.
