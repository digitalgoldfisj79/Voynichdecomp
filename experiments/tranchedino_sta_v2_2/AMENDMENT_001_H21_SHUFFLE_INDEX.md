# Amendment 001 — H21 shuffle segment indexing

Date: 2026-08-09
Status: prospective execution correction; frozen before any H21 cipher/language score.

The initial `stage_h21.py` encoded each retained segment in the deterministic shuffle seed as `line_index:subindex`. The inherited v2.1 protocol specifies `SHA256("TRANCHSTA21Hshuffle::<rep>::<folio>::<segment_index>")`.

H21 has not been scored. This amendment makes the implementation literal: `segment_index` is the zero-based sequential retained-segment index within each folio, in the order returned by the frozen RF parser. The same per-folio indexing is used when a bucket subset is scored, so each physical retained segment receives the same seed identity in the whole-panel and bucket calculations.

No scientific quantity changes: RF source/hash, protected H21/C21 split, K36 vocabulary, canonical T20 map, Paduan model, absolute score floor, 200 shuffle count, within-segment permutation null, bucket assignment, quantile definition, decision thresholds, replication rules and C21 seal are unchanged.

The pre-amendment H21 runner was never executed and generated no H21 score. This is an implementation/protocol alignment, not a response to target data.
