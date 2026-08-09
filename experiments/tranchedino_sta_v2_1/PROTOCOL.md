# Tranchedino × STA v2.1 — Clean continuation protocol

Date: 2026-08-09
Namespace: `TRANCHSTA21`
Parent v2.0 diagnostic head: `e3a7362adf53d514dd16d37a1a812073357cbd8f`.

## Scope

v2.1 changes one thing only: optimizer search depth/convergence control, prospectively justified by T20-only diagnostics. The historical f069v key geometry, RF K=36 vocabulary rule, Paduan model, absolute positive-control floor, null design and inference thresholds are inherited unchanged from v2.0.

The accidentally exposed v2.0 H20 folios are permanently excluded from v2.1 inference.

## Protected panels

Reconstruct the v2.0 RF split exactly. Take only the 68 never-scored v2.0 C20 folios. Sort that set by SHA-256 of `TRANCHSTA21split::<folio>` and divide 50/50:

- H21: first 34 folios — new first adjudication;
- C21: remaining 34 folios — sealed confirmation.

No language/cipher score from either panel may be generated before the binding T20 fit gate passes. C21 remains unscored unless H21 and cross-transliteration replication pass.

## Binding T20 optimizer

Rerun T20 from scratch under four fresh ensembles `G,H,I,J`, each with 72 restarts and the unchanged exact f069v multiplicity profile and pair-block local polish.

T20 convergence requires:

1. the highest objective among the four ensembles is attained by at least two ensembles within `1e-7` nats/retained symbol;
2. among those top-objective ensembles, occurrence-weighted map agreement is >=0.90;
3. the canonical map is the lexicographically smallest numerical-map vector among the agreeing top-objective maps (normally they should be identical).

If this fit gate fails: `V2.1 T20 FIT NOT CONVERGED`; no H21 score is admissible.

The previously observed C/E diagnostic map is not inserted, seeded, or privileged in v2.1.

## H21 gates

With the canonical T20 map frozen, H21 must pass all inherited v2.0 gates:

- retained K36 coverage >=0.97;
- H21 fixed-map Paduan quadgram score >= `-2.3672276834921244` nats/scored event (5th percentile of the 12 qualified fresh historical controls);
- H21 score >99th percentile of 200 deterministic within-line order-shuffle nulls;
- all four H21 folio buckets have observed-minus-null-median >0.

H21 bucket: first digest byte of SHA-256(`TRANCHSTA21Hbucket::<folio>`) modulo 4.
Shuffle seed: SHA-256(`TRANCHSTA21Hshuffle::<rep>::<folio>::<segment_index>`).

If H21 fails any gate: `NO TRANCHEDINO-STA ALPHABETIC SIGNAL`; C21 remains sealed.

## Replication and confirmation

Only after H21 passes, apply the same fixed RF T20 map without refitting to matching H21 loci in IT2a, ZL3b and GC2a_1 STA streams. At least 2/3 must exceed their 95th-percentile within-line shuffle null and lie within 0.05 nats/retained symbol of RF H21.

Only if replication passes may C21 be scored. C21 uses the same absolute floor and 200-shuffle q99 gate, all four `TRANCHSTA21Cbucket` bucket deltas must be positive, and at least 2/3 independent transliterations must replicate at p<=0.05.

Positive verdict: `CONFIRMED TRANCHEDINO-STA ALPHABETIC SIGNAL`.

No readable plaintext or word-level decode may be emitted before C21 confirmation.
