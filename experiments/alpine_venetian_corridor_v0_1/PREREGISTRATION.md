# PREREGISTRATION v0.1

Frozen: 2026-08-08

## Primary claim under test

`corridor_core` manuscripts in the frozen period have higher cross-family VMS affinity than matched non-corridor manuscripts.

## Selection firewall

The candidate cohort is frozen before any corridor-vs-control similarity comparison. Candidate inclusion may use only:

- catalogue date;
- authoritative production/origin geography;
- manuscript status;
- neutral content/genre;
- presence or inspectability of illustration;
- availability of a facsimile/IIIF image.

Forbidden selection variables:

- VMS similarity score;
- nearest-neighbour rank to any VMS object;
- phrases such as `Voynich-like` or citations asserting resemblance;
- post hoc knowledge that a manuscript produces an interesting match.

Existing comparanda discovered through Voynich-related work must independently satisfy the neutral rule and retain `discovery_bias=legacy_voynich_context`.

## Frozen geography labels

`corridor_core`, `corridor_buffer`, `control_lombardy`, `control_tuscany`, `control_bavaria_swabia`, `control_east_alpine`, `unresolved`.

`corridor_buffer` never silently becomes core in the primary test.

## Frozen period labels

`A_primary=1390:1450`, `B_antecedent=1350:1389`, `C_reception=1451:1475`, `D_late_reception=1476:1500`.

## Frozen models / representation policy

Image backbone family: **DINOv3**. The preferred high-capacity frozen backbone is `dinov3_vit7b16` / LVD-1689M when compute permits; a smaller DINOv3 model may be used only as a predeclared compute substitution and must be applied identically to VMS, corridor and controls. Exact repository/model revision and weights hash are recorded in the run row before embedding begins.

No fine-tuning on VMS labels, corridor labels or control labels.

Structured-description arm: one fixed prompt/schema for all corpora. The description prompt must explicitly forbid manuscript/place/school identification and VMS comparison.

Geometry arm: deterministic class-specific features from masks/boxes/landmarks.

## Frozen image transformations

Each inferential crop produces:

1. `gray_bgdiv_v1`: background-flattened greyscale crop;
2. `inkmask_v1`: thresholded/edge structural rendering where robust;
3. `rgb_norm_v1`: normalized RGB retained only as sensitivity/descriptive arm.

Raw page RGB is never a primary similarity representation.

## VMS reference freeze

Primary VMS references are reviewed/manual objects. If a class lacks enough reviewed references, that class is marked `reference_underpowered`; unreviewed detections do not silently enter the primary endpoint.

## Matching

For every eligible corridor manuscript, controls are drawn without replacement where possible using this priority:

1. same time bin;
2. overlapping broad genre/content tags;
3. same substrate if known;
4. comparable facsimile completeness/usable image count;
5. different holding institution where possible.

If multiple controls tie, deterministic ordering uses SHA-256 of `candidate_key || 20260808`.

Target: 3 controls per corridor manuscript when available; minimum 1.

## Score aggregation

Object comparisons are clustered by manuscript. Crop count never becomes sample size.

Within `(manuscript, class, arm)`, summarize VMS similarity using a robust top-k statistic, where:

- `k = min(5, number_of_usable_comparandum_objects)`;
- use the median of the top k object-level best-match similarities;
- calibrate to the empirical non-VMS null for the same class/arm.

Class scores are averaged within an arm. Arms receive equal weight in the composite after null calibration. A missing class/arm is omitted, not set to zero.

## Primary test

- unit: manuscript;
- statistic: mean corridor-minus-control composite difference within matched strata;
- null: permutation of corridor/control label within strata;
- permutations: 100,000;
- RNG seed: 20260808;
- two-sided p-value;
- positive direction required;
- alpha: 0.01.

## Family convergence

To call Tier 2 or above:

- >=3 independent families positive;
- >=2 family tests BH-FDR q<0.05;
- no single manuscript contributes >35% of the total corridor-control effect magnitude;
- leave-one-manuscript-out retains positive effect in >=90% of deletions.

## Confound gate

Grouped manuscript/institution classifier on final pixel embeddings:

- AUC <=0.65 PASS
- (0.65,0.70] CAUTION
- >0.70 FAIL image arm

A failed image arm is removed from the primary composite without replacement by another pixel model.

## Coverage gate

Primary inference requires:

- >=12 verified illustrated corridor-core manuscripts across A+B;
- >=8 matched control manuscripts in at least two control ecologies;
- usable image coverage documented for all included manuscripts.

Failure => `UNDERPOWERED`, regardless of interesting examples.

## Stopping rules

Stop and issue a result if any occurs:

1. full frozen census exhausted for reachable catalogues and coverage gate passes;
2. full frozen census exhausted and coverage gate fails -> UNDERPOWERED;
3. confound gate fails all pixel arms and non-pixel arms are insufficient -> NONRESOLVING;
4. primary H1 rejects under frozen rule -> proceed only to preregistered sensitivity checks, then stop;
5. primary H1 does not reject -> run preregistered sensitivity checks for diagnosis, then stop; no new primary feature family may be added.

## Prohibited post hoc moves

- widening geography because a near-miss looks good;
- moving a 1451+ manuscript into the primary bin;
- choosing a model after seeing corridor/control separation;
- excluding an inconvenient control except for a predeclared QA failure;
- adding feature weights that improve H1;
- treating many crops from one manuscript as independent n;
- interpreting holding location as production location;
- claiming a provenance narrower than the evidence tier permits.
