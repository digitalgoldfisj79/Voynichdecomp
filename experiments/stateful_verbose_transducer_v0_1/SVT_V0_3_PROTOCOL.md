# SVT v0.3 — manuscript-scale stateful verbose qualification

Frozen: 2026-08-21, before v0.3 synthetic results.

## Basis

SVT v0.2 failed its binding 192-character Gate A (mean recovery 34.9%, median 24.5%, 0/8 >=85%). It remains a failed gate and is not reinterpreted.

A separately registered post-failure scale diagnostic, using fresh keys and the unchanged v0.2 deterministic factorised solver with true mode/period, showed a strong information-scale transition:

- 384 chars: mean 47.4%;
- 768 chars: mean 86.7%;
- 1536 chars: mean 98.89%, minimum 98.05%, 4/4 >=85%.

The target manuscript contains orders of magnitude more text than 1,536 plaintext units. v0.3 therefore tests whether the combined mechanism is recoverable at a scale where the synthetic state-specific mappings are identifiable, rather than relaxing recovery quality.

## Mechanism and solver

Unchanged from v0.2:

- factorised stateful substitution: one shared fresh base key plus sparse state-local swaps;
- periodic and line-reset modes;
- periods 2–12;
- no circular symbol order;
- variable-length 1–3 glyph verbose renderer remains the downstream layer;
- deterministic whole-sequence coordinate optimiser for shared/global and state-local swaps;
- no Voynich loader.

## Fresh instance discipline

v0.3 development uses fresh synthetic replicate IDs offset by `+5000`, not used in v0.1/v0.2 smoke, Gate A, or scale diagnostics.

Locked testing, if reached, will use the untouched `test` split and a separate `+7000` offset.

## Gate A3 — true-structure key recovery

Data:
- German pinned development corpus;
- 8 fresh trials, 4 periodic + 4 line-reset;
- plaintext length 1,536;
- true mode and true period supplied;
- shared base key and all state-local swaps hidden.

Required, deliberately stricter than v0.2:
- mean plaintext recovery >= 0.95;
- median >= 0.97;
- minimum >= 0.85;
- at least 7/8 trials >= 0.90.

Failure closes v0.3 before any blind-structure or segmentation work.

## Gate B3 — blind structure on true heads

May be built/run only if A3 passes. True verbose unit heads are supplied, but mode, period and all keys are hidden.

Required on a new fresh set of 8 development trials at length 1,536:
- mean recovery >= 0.90;
- median >= 0.95;
- at least 7/8 >= 0.85;
- mode accuracy >= 7/8;
- period accuracy >= 6/8.

A deterministic structure-screening stage may be used only if frozen and calibrated before B3; no truth-selected shortlist.

## Gate C3 — hidden segmentation

May be designed only if A3 and B3 pass. The v0.1 local-surprisal segmenter is closed as inadequate and is not reused as the primary method.

The replacement must infer variable-length boundaries jointly with recoverable latent heads and must pass known-answer boundary/recovery gates on new synthetic instances before any target access.

## Locked gate and target

A fresh multilingual/language-blind locked test with hostile NONFACT and SHUFFLED controls remains mandatory before target access. The target runner remains sealed until that locked gate passes.

No Voynich data may be scored, decoded or used to choose hyperparameters during v0.3 synthetic development.