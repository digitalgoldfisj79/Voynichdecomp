# Voynich MRC — bifolium packet source-vs-production discriminator result — 2026-10-04

- key: `voynich_mrc_packet_source_production_discriminator_result_20261004`
- status: active
- created: 2026-10-04T11:59:39.120913+00:00
- updated: 2026-10-10T16:27:25.988447+00:00

---


# BIFOLIUM PACKET SOURCE-vs-PRODUCTION DISCRIMINATOR — RESULT
Date: 2026-10-04
Preregister:
voynich_mrc_packet_source_production_discriminator_preregister_20261004

## FIREWALL
NO P70 FEATURES, P70 QUIRE MAPPING, P70 GRAMMAR OR P70 RESULTS USED.

Independent physical metadata:
public.vms_boundary_localisation_paragraph_v01
frozen export commit:
80d5c9778cb27e3dd20b049ab8804a7bb8832fd2

Fields used:
folio, actual quire, true bifolio id, Currier, Davis hand, word/line count.

MRC script:
research/packet_source_production_discriminator_20261004.py
corrected commit:
8fe167f6072f2f7ab096ad50963ec758a60e0f5b

HF job:
6ac23f40404719ba3764da7d
Status: COMPLETED.

An earlier execution using globally keyed bifolio ids was invalid because bifolio ids repeat by quire. It was discarded before interpretation. Corrected key is (quire,bifolio).

## FROZEN TEXT MODEL
Current MRC/K12 only.
Frozen packet donor ridge L2=10.
Frozen donor strength lambda=.25.
Heldout folds0/1 only.
40 heldout numbered folios.

## PRIMARY SAME-QUIRE CONTROL
30 directional target leaves had an available heldout same-quire nonmate control.
15 true-bifolium blocks.

Control tiers:
22 directions T1:
same quire + same section + same Currier + same Davis hand as true mate.
8 directions T4:
same quire + same section.
No lower tiers used.

Mean directional true-mate advantage:
+0.00540639 bits/event.

After aggregating both directions inside each physical bifolium:
B=15 blocks.
Exact exhaustive 2^15=32768 sign flips.
Observed mean block advantage:
+0.00540639 bits/event.
Block SD=.01170627.
mean/SD=.46184.
one-sided exact p=.0473938.

PREREGISTERED PASS required p<.01.
DECISION: FAIL / METRIC DOES NOT RESOLVE mate-specific advantage at the preregistered gate.

## NEAREST SAME-QUIRE CONTROL
True mate vs numerically nearest same-quire nonmate.

15 bifolium blocks.
Mean advantage:
+0.00543122 bits/event.
Block SD=.0121850.
mean/SD=.44573.
Exact p=.0534973.

DECISION:
METRIC DOES NOT RESOLVE.
True mate is not demonstrably better than the nearest same-quire production neighbour.

## ALL SAME-QUIRE RANK
30 target directions.
True mate rank 1:
13/30.
Median rank=2.
Mean rank=1.8333.

Descriptively better than many same-quire alternatives, but no preregistered rank null was used for promotion.

## GENERIC QUIRE EFFECT
For targets with adequate matched cross-quire controls:
16 directions / 8 bifolium blocks.

Same-quire nonmate advantage over same-section/same-Currier/same-hand cross-quire donors:
+0.00217813 bits/event.

Exact sign-flip:
p=.371094.
mean/SD=.12220.

DECISION:
METRIC DOES NOT RESOLVE a generic quire effect in this reduced matched subset.

This does NOT mean quire is absent; it means this specific matched test is underpowered/unresolved.

## CROSS-REGISTER SENSITIVITY
0 retained target directions had true mates differing in Currier and/or dominant Davis hand.
Therefore this discriminator cannot test cross-register persistence.
No claim licensed.

## INTERPRETATION
The earlier broad packet result remains valid relative to same-section/hand nonmates:
true physical mate advantage +.0022457 bits/event, matched p=.0004998.

However, once controls are forced into the SAME ACTUAL QUIRE/local production neighbourhood, the true-mate advantage falls to an unresolved p=.0474 under the stricter preregistered p<.01 gate.

Nearest same-quire comparison likewise fails at p=.0535.

Therefore:
1. Do NOT promote the residual packet signal as source/exemplar-specific.
2. Do NOT claim a proven bifolium-specific source packet.
3. The data are compatible with a broader local production/quire context accounting for much of the earlier packet signal.
4. A mate-specific residual remains descriptively positive, but is unresolved at the project gate.
5. Generic quire effect itself is not separately resolved in the reduced matched subset, so the correct status is CONFOUNDED/UNRESOLVED, not "quire proven."
6. The source-vs-production distinction remains open, but current evidence shifts weight toward production-neighbourhood confounding relative to the stronger source-packet interpretation.

## ARCHITECTURAL CONSEQUENCE
Retain:
- FORM as fast local process;
- LINE_ENTRY as separate physical-line controller;
- current MRC SELECT baseline and weak line repertoire coupling.

Downgrade:
- SLOW BIFOLIUM/SOURCE CONTEXT as a promoted independent layer.

Replace with:
SLOW LOCAL PRODUCTION / QUIRE / PACKET CONTEXT — unresolved scale and cause.

No new precept is added.

## NEXT, IF CONTINUED
A stronger source-vs-production discriminator would require one of:
- out-of-quire but historically reconstructed source-linked packets;
- cross-hand or cross-Currier true bifolia;
- independent material/ink/ductus production covariates to partial out production state;
- codicologically reconstructed writing order/work sessions.

Without such independent production covariates, further text-only same-quire tuning risks overfitting the same confound.


## 2026-10-04 TOUCHING-FACE DISCRIMINATOR
Completed result: voynich_touching_face_discriminator_result_20261004_v01

Critical direct same-bifolium discriminator:
touching face vs non-touching face from same bifolium:
effect +0.00229000 bits/event; sign-flip null SD .00289771; effect/null-SD .7903; p=.229492.
THE METRIC DOES NOT RESOLVE a contact-specific advantage.

Do not interpret folded-face contact as the mechanism behind the residual packet/bifolium signal. Cross-bifolium touching remains underpowered (one usable edge), so zero contact effect is not established.

