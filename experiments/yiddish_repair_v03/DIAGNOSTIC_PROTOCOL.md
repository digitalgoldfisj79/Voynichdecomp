# Known-key diagnosis and constrained repair

## Retracted findings

- The post-v0.1b claim that failure was merely calibration is retracted: v0.2 also failed correct-language recovery. The cause of those recovery failures is not yet established.
- All v0.1/v0.1b/v0.2 failures remain controlling; no threshold change can rewrite them.

## Scope and pre-analysis checks

Use the hash-verified v0.2 public artifact and final committed outputs/truth. Diagnose all families, including passing controls, after blind outputs were committed. No target access. Old exposed cases are development only, never fresh confirmation.

Order: circularity (oracle diagnostic only), leakage (no training on diagnostic audit), confounds (same model/representation), matched nulls (retain each original random-map null), fairness (both language models), degeneracy (finite scores, exact round trip), representation (source hashes unchanged), decision fragility (no threshold rescue), completeness (all keys/families).

Tests: independently reconstruct fit and audit scores for returned and true keys; compare true-key score with the strongest single key-swap neighbour under the same correct-language model; inspect recovered character confusions and errors by length. Oracle recognition uses the same plaintext under both models and is diagnostic, not a deployable qualification. Record raw contrasts and null dispersion together. Verify swap-delta against full rescoring before interpreting neighbours.

Repair decision: a missed better true key motivates search changes; incorrect keys scoring better than truth motivate objective/model or representation investigation. No model change is accepted merely because it repairs these cases. Fresh unseen family qualification and external confirmation remain required before Voynich. Checkpoint each family atomically with pickle and JSON.
