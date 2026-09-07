# VMS edge forensics + orientation v0.1 — PREREGISTERED BEFORE CANDIDATE INSPECTION

Date: 2026-09-07
Protocol ID: `vms_edge_forensics_orientation_20260907_v01`
Parent protocols:
- `vms_blind_bifolium_seriation_20260907_v01`
- `vms_seriation_metadata_residual_20260907_v01`
- `vms_seriation_residual_retention_20260907_v01`

## Scientific question
Can any of the five metadata-residual bifolium-pair edges be upgraded from a reproducible text-state similarity to an independently corroborated **production-neighbour relationship**? If yes, can any validated neighbour relation be **oriented** without using present folio order as ground truth?

## Frozen candidate set
No new text-derived candidate may enter v0.1 after this file is committed.

Primary cross-quire candidates:
- E1: `q01_b3_6` ↔ `q03_b17_24` = (f3,f6) ↔ (f17,f24), residual support 12/12.
- E2: `q06_b42_47` ↔ `q01_b1_8` = (f42,f47) ↔ (f1,f8), residual support 12/12.

Secondary candidates:
- E3: `q05_b36_37` ↔ `q03_b19_22` = (f36,f37) ↔ (f19,f22), residual support 7/12.
- E4: `q13_b76_83` ↔ `q13_b77_82` = (f76,f83) ↔ (f77,f82), residual support 12/12.
- E5: `q13_b75_84` ↔ `q13_b78_81` = (f75,f84) ↔ (f78,f81), residual support 10/12.

The pharmaceutical q15/q19 triangle and Q9–Q10–Q11 are explicitly excluded as ordering evidence in v0.1 because they fail metadata residualisation.

# STAGE F — FORENSIC EDGE VALIDATION

## F0. Freeze / leakage rules
Before inspecting candidate-specific physical evidence:
1. Candidate list and edge strengths above are frozen.
2. No additional textual feature may be invented to rescue an edge.
3. Folio/current-quire labels may be used only to retrieve images/codicological records and to define matched controls; current numerical adjacency is never positive evidence.
4. A source that discusses one candidate pair because of previously noticed textual similarity is not independent corroboration unless the physical observation itself is documented independently of that similarity.
5. Negative or unavailable evidence stays negative/unavailable; absence must not be converted to weak support.

## F1. Text robustness deletion tests
Purpose: ensure the retained edge is not driven by one feature family or one transcription.

For every E1–E5:
- leave one discovery family out: LEX, C3, SHAPE;
- leave one transcription out: ZL, RF, IT, GC;
- recompute residual consensus with the frozen pair set only;
- no retuning of k or thresholds.

Pass `TEXT_ROBUST` only if:
- edge remains consensus under at least 2/3 feature-family deletions; AND
- remains consensus under at least 3/4 transcription deletions.

Otherwise the edge is demoted to `REPRESENTATION_DEPENDENT` and cannot be physically promoted in v0.1.

## F2. Held-out text validators
Frozen validators, not used in discovery:
- RARE: overlap of globally rare word types, with the previously used rare threshold frozen by parent assay.
- LAYOUT: line-token-count / line-density distribution.

Matched null hierarchy, in this order:
A. same Currier + hand + broad illustration/section class where >=20 admissible alternative pairs exist;
B. same Currier + hand where >=20 alternatives exist;
C. same Currier + broad illustration/section where >=20 alternatives exist;
D. all complete bifolia excluding the candidate endpoints.

The first level meeting n>=20 is used and recorded; lower levels may be sensitivity checks only.

Held-out corroboration passes if at least one validator has |effect|/null SD >=2 and p<=.01, with effect and null SD reported together. If <2: `the metric does not resolve this`.

## F3. Physical/codicological evidence families
Candidate-specific inspection is limited to these predeclared families:

P1. parchment/material geometry
- dimensions/aspect after scan normalisation;
- edge contour/cropping-independent leaf geometry;
- hair/flesh pattern only where catalogue or imagery supports confident assignment;
- ruling/pricking pattern where visible/documented.

P2. stains/damage/wear
- stain profile/contour/intensity;
- tears, losses, holes, abrasion, exposed-edge wear;
- spill chronology where already calibrated;
- matched top/bottom or same-region control required for automated image metrics.

P3. transfer/offset evidence
- pigment or ink offset, rubbing/transfer marks;
- only directional if donor/receiver morphology is distinguishable independently.

P4. sewing/folding/collation
- old sewing holes, tackets, fold deformation, quiremarks;
- current sewing alone is not evidence of earlier neighbourhood;
- absence of visible old holes is not negative unless visibility/sensitivity is established.

P5. cross-surface composition / layout
- drawings, pipes, lines, diagrams or other compositions crossing a fold/sheet;
- establishes internal bifolium function, not inter-bifolium adjacency unless a second sheet is physically implicated.

P6. correction / production dependency
- cross-sheet corrections, continuation instructions, catchword-like cues, unfinished-to-finished dependencies, explicit scribal continuation;
- direction may be licensed only if A must precede B under a documented production dependency.

P7. documentary codicology
- Davis/Layfield/Yale or other primary/peer-reviewed codicological observations specific to these leaves;
- later speculative reconstructions are contextual only unless the underlying physical observation is independently inspectable.

## F4. Physical matched controls
A quantitative physical metric may be used only if it has a matched null defined before candidate evaluation.
Preferred nulls:
- same acquisition/scanning tranche and same margin/leaf side;
- same current quire when enough leaves exist;
- same parchment-side orientation where known;
- all eligible leaves only as fallback.

A physical channel is `POSITIVE` only if:
- effect magnitude >=2 matched-null SD AND p<=.01;
- calibration/known-positive pair was not the candidate itself;
- matched control channel (e.g. bottom edge for upper stain) does not show an equal-or-larger effect;
- result survives leave-one-cue-family-out if a composite score is used.

Otherwise: `the metric does not resolve this`.

## F5. Edge promotion rule
An edge becomes `CANDIDATE_PRODUCTION_NEIGHBOUR` only if all are true:
1. `TEXT_ROBUST` passes;
2. at least one held-out text validator passes OR the edge has an independent high-confidence documentary physical observation;
3. at least one independent physical/codicological family P1–P7 is positive/compatible in a way that discriminates the candidate from matched alternatives;
4. no hard physical constraint contradicts the relation.

A text-only edge remains `TEXT_STATE_NEIGHBOUR`.

## F6. Strong falsifiers
Demote/kill an edge if any occur:
- one discovery family or one transcription uniquely carries it under deletion;
- held-out validation is <2 SD and no independent physical corroboration exists;
- a hard physical observation makes neighbourhood impossible at the relevant chronology stage;
- candidate is no more physically similar/compatible than matched alternatives under all testable cue families.

# STAGE O — DIRECTION / ORIENTATION (CONDITIONAL)

Stage O remains sealed unless at least one Stage-F edge is promoted.

## O0. Principle
Symmetric similarity cannot establish sequence. Direction requires an asymmetric observable that is qualified on known-order/control material before target opening.

## O1. Frozen candidate directional feature families
No other family may be added after Stage O opens.

D1. repertoire introduction asymmetry
- estimate whether rare/new types concentrated in B are predictable from A more than the reverse;
- vocabulary is frozen globally; no candidate-specific feature selection.

D2. conditional predictive asymmetry
- train a simple smoothed unigram/character model on unit A and score B, versus train on B and score A;
- normalise by target entropy/length;
- this is descriptive directionality only until controls qualify it.

D3. correction/dependency direction
- only from P6 physical evidence; if a correction/continuation physically requires A before B, direction is deterministic evidence rather than a statistical text score.

D4. transfer/offset direction
- only if donor/receiver can be independently identified from physical morphology.

## O2. Calibration controls
Before opening Voynich direction:
- use at least two known-order manuscript/text controls with unit-resolved transcription;
- one must be medieval manuscript material where folio/quire order is independently secure;
- the second may be a continuous public-domain text segmented into matched unit lengths, but cannot alone qualify the instrument.

For each control:
- hide true direction;
- predict direction for adjacent known units;
- report balanced accuracy and edge-level calibration;
- require >=0.75 balanced accuracy on the manuscript control and >=0.70 on the second control, with 95% bootstrap lower bound >0.5 on the manuscript control.

If calibration fails: `DIRECTION INFERENCE NOT QUALIFIED`; Stage O target stays sealed.

## O3. Target direction decision
For a promoted production-neighbour edge:
- D1 and D2 must agree on direction in >=3/4 transcriptions and each exceed 2 matched-null SD;
- OR one independent physical deterministic direction (D3/D4) may orient the edge directly;
- disagreements remain `UNDIRECTED`.

Never orient by present folio numbering, present quire sequence, or similarity-gradient optimisation.

# STAGE G — GRAPH INTEGRATION

Only promoted Stage-F edges enter the production graph.
Only Stage-O-qualified directions become arrows.

Outputs:
- isolated validated pairs;
- connected undirected components;
- directed partial-order fragments where licensed;
- no forced Hamiltonian path.

A 3+ node ordering seed requires two or more promoted edges sharing a node and at least one licensed direction, with no physical contradiction.

# Execution order
1. Persist this preregistration and SHA.
2. Run F1 deletion tests on all five edges.
3. Run F2 held-out validators on all five edges.
4. Conduct candidate-blind physical cue extraction where feasible, using all eligible leaves before inspecting candidate ranks.
5. Search primary codicological literature for candidate leaves, recording positive, negative and unavailable evidence.
6. Apply F5 promotion rule.
7. If zero edges promoted: stop Stage O; bank failure.
8. If >=1 promoted: calibrate Stage O on controls; do not open Voynich directions until calibration passes.
9. Integrate only licensed edges/directions into the canonical topology graph.

# Reporting rules
- Every quantitative comparison: effect and null SD in the same sentence.
- Ratio <2: lead with `the metric does not resolve this`.
- Retractions/failures appear at the top of every closeout.
- No candidate-specific threshold changes after this commit.
