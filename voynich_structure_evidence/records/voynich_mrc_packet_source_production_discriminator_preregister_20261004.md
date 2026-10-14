# Voynich MRC — bifolium packet source-vs-production discriminator preregistration — 2026-10-04

- key: `voynich_mrc_packet_source_production_discriminator_preregister_20261004`
- status: active
- created: 2026-10-04T11:55:00.961986+00:00
- updated: 2026-10-10T16:27:25.988447+00:00

---


# BIFOLIUM PACKET SOURCE-vs-PRODUCTION DISCRIMINATOR — PREREGISTRATION
Date: 2026-10-04

## FIREWALL
NO P70 FEATURES, P70 QUIRE MAPPING, P70 GRAMMAR OR P70 RESULTS ARE USED.

Text model is only the current frozen MRC/K12 prospective stack:
- ED-free SELECT baseline;
- page/paragraph/recent reinforcement;
- K4 opener-house-conditioned line repertoire;
- K12 residual donor tilt used in the already-frozen packet-transfer test.

Physical/control metadata are independently frozen from:
public.vms_boundary_localisation_paragraph_v01
export commit 80d5c9778cb27e3dd20b049ab8804a7bb8832fd2
fields only:
folio, actual quire, true bifolio id, Currier, Davis hand, word/line count.

## PARENT RESULT
On untouched folds0/1, true physical mate donor:
+0.0060613 bits/event vs frozen baseline.
+0.0022457 bits/event over same-section/hand matched nonmates.
matched p=.0004998.
This test asks whether that advantage survives LOCAL PHYSICAL PRODUCTION controls.

## TARGET / DONOR
Heldout folds0/1 only.
Each numbered leaf aggregates its recto+verso residual SELECT events.
True donor = the other numbered leaf sharing the independently recorded true bifolio id.
No traversal direction.

## PRIMARY SAME-QUIRE CONTROL
For each target leaf:
candidate controls are other heldout leaves in the SAME ACTUAL QUIRE, excluding target and true mate.

Use the strongest available metadata stratum, frozen in this order:
T1 same quire + same section + same Currier + same Davis hand as true mate;
T2 same quire + same section + same Currier;
T3 same quire + same section + same Davis hand;
T4 same quire + same section;
T5 same quire.

Within the highest nonempty tier, rank candidate nonmates by a target-blind physical matching distance:
| numeric leaf-distance(candidate,target) - numeric leaf-distance(mate,target) |
plus
| log(candidate scored-token count / mate scored-token count) |.

Use the best up to 3 nonmates; ties by folio number.
No textual predictive score enters control selection.

Primary per-target contrast:
mean matched-nonmate predictive codelength - true-mate predictive codelength.
Aggregate the two target directions within each true bifolium before inference.

Primary inference:
exact exhaustive sign-flip over the available bifolium-block contrasts (all 2^B signs where feasible).
PASS if one-sided p<.01 and mean advantage >0.

## SECONDARY CONTROLS
A. Nearest-neighbour control:
true mate vs numerically nearest same-quire nonmate.

B. All-same-quire rank:
rank true mate against every available same-quire nonmate donor.

C. Quire effect:
compare average same-quire nonmate prediction against same-section/same-Currier/same-hand cross-quire donors.
This distinguishes a generic quire context from a mate-specific increment.

D. Cross-register sensitivity:
separately report true-mate advantage for bifolia whose two leaves differ in Currier and/or dominant Davis hand, if >=3 such bifolia are available.
No claim if underpowered.

## INTERPRETATION
True mate does NOT beat same-quire controls:
classify prior packet result primarily as quire/local-production context.

True mate beats same-quire controls:
generic quire/local proximity is insufficient; promote a BIFOLIUM-SPECIFIC working context.

Even on PASS, source/exemplar is strengthened but NOT proven, because a bifolium-specific production/work-session mechanism remains possible.

Source interpretation is strengthened further only if:
- mate-specific advantage survives cross-hand/Currier bifolia where available;
- true mate beats nearest local neighbours;
- same-quire generic effect alone cannot explain it.

No new token rule or p70 construct may be introduced.

