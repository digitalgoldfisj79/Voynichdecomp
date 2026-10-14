# Latent transform discriminator — encoded vs surface-generated — 2026-09-28

- key: `voynich_latent_transform_discriminator_20260928`
- status: active
- created: 2026-09-28T11:05:54.609415+00:00
- updated: 2026-10-10T16:27:25.988447+00:00

---

# Latent transform discriminator — 2026-09-28

## RETRACTED / DOWNGRADED
- SG64/top-8 realization families do NOT currently support a claim that a latent lexical/plaintext layer has been recovered.
- The Currier A/B JSD collapse remains real but is insufficient: on a fresh Stars early/late split SG64 does not beat family-size-matched random compression.
- A correct latent collapse was predicted to reduce P1/P3/E-2 boundary anomalies. It fails: P1 becomes substantially more anomalous; P3/E-2 do not resolve as reduced.
- Absolute held-out predictability does not improve under top-8; top8+gallows is unresolved by folio bootstrap.
- This rejects a state-free/simple latent-root recovery model, NOT every sufficiently stateful cipher/transduction model.

## Frozen transform
SG64 conservative top8:
INS:d:PRE_FINAL:R=y
SUB:e>o:INTERNAL
SUB:a>o:PRE_FINAL
SUB:d>e:PRE_FINAL
INS:o:INTERNAL:R=d
SUB:m>r:FINAL
SUB:l>o:INITIAL
SUB:d>k:INITIAL

Rebuilt exact connected-component mapping:
SG104 primary types 3121 ->2493 latent families (-20.12%); 60.28% token mass in non-singletons.
Persistent tables:
vms_latent_top8_edges_v01
vms_latent_top8_map_v01
vms_latent_top8_token_work_v01
vms_latent_stars_entries_v01
All new tables RLS-enabled.

## Gate 1: Stars positional de-anomalization — FAIL
Full ZLZI marked Stars spans, 264 entries >=18, exact tokens; unmapped token=self.

P1 raw -4.52 pp / null SD .91 pp = -4.98 SD.
P1 latent -9.46 /1.38 = -6.87 SD.
Latent-minus-raw -4.94 pp / paired null SD .99 pp = -5.01 SD: WRONG WAY, strongly.

P3 raw +5.94/2.13=+2.79 SD.
P3 latent +6.50/2.44=+2.66 SD.
Change +.57/1.53=+.37 SD unresolved.

E-2 raw +5.34/2.46=+2.17 SD.
E-2 latent +6.13/3.04=+2.02 SD.
Change +.79/2.08=+.38 SD unresolved.

## Gate 2: fresh Stars early/late cross-context stability beyond compression — FAIL
SG104 primary:
raw JSD .309188 bits.
top8 JSD .247551 bits (-19.93%).

Matched null:
2000 deterministic random collapses preserving exact 3121-type ->2493-family size multiset.
random mean JSD .245637, SD .001928, 95% roughly [.241943,.249341].
Top8 minus random mean +.001914 / .001928 = +.99 SD. Lower is better: top8 is slightly worse than random, unresolved.

Historical positive retained:
Currier A/B raw .46057 -> top8 .32460 (-29.52%), excess over component-size matched random collapse 14.17 SD.
This positive does not generalize to fresh Stars early/late stability.

## Gate 3: held-out absolute predictability — FAIL
Reuse existing SG66 frozen machinery.
Exact validation baseline 10.271025 bits/token.
Top8 actual 10.275393; gain -0.004367 bits/token / matched-random null SD .004787.
Matched-random mean gain -.041227 across 500 nulls.
Top8 is specifically much less bad than random (historical 7.70 SD) but absolute prediction is WORSE than exact baseline.

Top8+gallows sensitivity:
gain +.006975 bits/token / folio-bootstrap SD .009126 = +.76 SD unresolved.
10k bootstrap q025 -.01071, q975 +.02514, fraction <=0 .2271.
Rare strata worsen: unseen -.18479; train hapax -.25560; freq2-4 -.17638 bits/token. Common forms improve.

## Gate 4: independent Stars folio halves — FAIL
Early <=f109:
P1 change -4.70 pp /1.73 = -2.72 SD (wrong way).
P3 change +2.95/2.47=+1.19 unresolved.
E-2 change -.76/2.76=-.27 unresolved.

Late >=f111:
P1 change -5.21 pp /.74 = -7.07 SD (wrong way).
P3 change -2.17/1.68=-1.29 unresolved.
E-2 change +2.54/3.16=+.81 unresolved.

Wrong-way P1 strengthening replicates independently.

## Audit
Circularity: operators were not selected for the present positional/JSD metrics.
Leakage: historical SG64 portability validation included Stars-B; therefore not pristine historical holdout. New metrics/folio split were not selection criteria and mostly falsify.
Compression confound: explicitly controlled by 2000 exact family-size-matched random collapses.
Measurement: exact ZLZI tokens, no PGCS. Same physical Stars spans before/after T.
Representation: top8 primary. Top8+gallows heldout sensitivity does not rescue.

## Decision
Frozen SG64 latent transform FAILS all four preregistered gates.

Licensed inference:
- Genuine surface realization families / Currier-conditioned state structure exist.
- They do NOT currently behave like removable noise around a simpler ordinary lexical sequence.
- Evidence shifts away from simple encoded-natural-language / latent-root collapse and toward a structurally real surface-generative or stateful-transduction layer.
- Cannot distinguish created/formal surface system from natural-language plaintext passed through a sufficiently stateful transducer.

Next only justified encoded-vs-created test:
prospective STATE-AWARE model inferred train-only from observable manuscript variables; predict exact surface realization on held-out folios, remove predicted state, and require improved absolute lexical/predictive regularity. Oracle/post-hoc states do not count.

