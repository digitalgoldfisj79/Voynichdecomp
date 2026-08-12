# VSN-B4-v1 Method Correction Log

## 2026-08-12 — source-stage runner v1 -> v1.1

Timing: correction made while target gate was CLOSED. At this point only B4A01 of 20 f.8v rows had an A/B transcription; no binding source controls and no new Voynich comparison had run.

The initial `run_b4_source_metrics_v1.py` was an implementation draft. Audit against the already-frozen B3 metric definitions found three incompatibilities:

1. v1 included string-boundary symbols in `H(next|prev)`; B3 does not.
2. v1 averaged positional entropies equally across positions; B3 weights each position by the number of characters contributing to it.
3. v1 classified the first/last quartile of an edit as prefix/suffix; B3 defines only the literal first character as prefix and literal final character as suffix; all other edit positions are internal.

The protocol requires direct comparability with the B3 targets, so these were corrected in `run_b4_source_metrics_v1_1.py` before target unlock.

The same revision also implements the two preregistered controls missing from the draft runner:
- C2 within-component character shuffle, preserving four syllable boundaries;
- C3 matched-length iid character marginal.

The original v1 file is retained unchanged for audit. v1.1 states `voynich_data_accessed=false` and remains blocked until all required source rows are resolved.

Reference B3 implementation: `experiments/voynich_semantic_notation_v1/state_gated_k2_v1/state_gated_k2_v1.py`, especially `surface_metrics()` and `edit_pairs()`.
