# Tranchedino × STA v2.2 — H21 adjudication result

Date: 2026-08-09
Protocol: v2.2 T20 qualification with inherited v2.1 H21 gates
T20 qualification result: `bcd371306cb7dcbad7d7df9b7293d1e30edca5cb`

## Provenance correction

The first H21 execution occurred in HF job `6a7882013e1f34a7e32c06b8`, immediately after the original H21 runner commit `4fbddc6022f4029c96d584a91f9e0479031369b6`. This job was discovered during the subsequent HF-history audit. It is therefore the **primary H21 adjudication**; H21 must not be described as pristine at the time of the later rerun.

The original runner used `line_index:subindex` in deterministic shuffle seeds rather than the protocol's per-folio sequential `segment_index`. That execution mismatch affects only which within-segment random permutation is selected. It does not affect the H21 panel, canonical T20 map, observed fixed-map score, historical-control floor, coverage, bucket assignment, or C21 seal.

The primary job produced:

- observed score `-2.491541751271494`;
- positive-control p05 floor `-2.3672276834921244` — **FAIL by -0.12431406777936971**;
- original-run shuffle q99 `-3.006862569040507` — order-null PASS;
- all four bucket deltas positive.

Because the preregistered absolute historical-control floor is a conjunctive gate and fails independently of the shuffle seed implementation, the primary H21 verdict is already negative.

Before this earlier job was discovered, the seed mismatch was prospectively aligned in `AMENDMENT_001_H21_SHUFFLE_INDEX.md`, and job `6a7887b73e1f34a7e32c0711` reran H21 with the literal per-folio segment-index seed. That run is now classified strictly as a **reproducibility check, not a virgin held-out adjudication**. It reproduced the identical observed score, coverage and bucket observed scores; its shuffle q99 was `-3.0075474513223406`, again passed, and all four bucket deltas remained positive.

No H21 result, original or rerun, unlocked C21 because the absolute historical-control gate failed. C21 was never scored; only its folio-list SHA-256 was emitted. IT2a, ZL3b and GC2a_1 replication streams were not scored. No decoded plaintext string or word-level reading was emitted or inspected.

C21 folio-list SHA-256: `6124810e506fd5dfbcfa6e9e6f445047b49d5e24af598a4ce9a0beb599cc890c`.

## Binding H21 result

- retained K36 coverage: **0.9949215710** (gate >=0.97: PASS)
- observed fixed-map Paduan quadgram score: **-2.4915417513 nats/scored event**
- qualified historical-control 5th-percentile floor: **-2.3672276835** (gate: FAIL by -0.1243140678)
- primary-run within-line shuffle median: **-3.0202403983**
- primary-run within-line shuffle q99: **-3.0068625690**
- primary-run observed minus q99: **+0.5153208178** (order-null gate: PASS)

Primary-run bucket deltas were all positive:

- bucket 0: +0.5379213224
- bucket 1: +0.5619100296
- bucket 2: +0.5159457552
- bucket 3: +0.4857857718

## Formal verdict

**NO TRANCHEDINO-STA ALPHABETIC SIGNAL.**

The fixed f069v-geometry map captures substantially more sequential Paduan-like structure than the within-line order-shuffle null, consistently across all four buckets, but its absolute H21 score is materially below the preregistered distribution of fresh, correctly generated Tranchedino/Paduan positive controls. The absolute historical-control gate is binding and is unaffected by the shuffle-seed provenance correction.

Under the frozen protocol, C21 and cross-transliteration replication remain sealed. No further restart escalation, remapping, threshold adjustment or direct 36-sign alphabetic confirmation is permitted in this programme.

This closes the direct full-STA fixed 36-sign f069v alphabetic mechanism. It does not close the separately scoped mixed-unit/null/geminate/nomenclator extension, whose model class and capacity differ materially and which requires its own prospective qualification before any Voynich target score.
