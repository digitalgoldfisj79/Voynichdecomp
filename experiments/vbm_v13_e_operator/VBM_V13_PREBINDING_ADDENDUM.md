# VBM v13 — pre-binding addendum

Date: 2026-09-02
Status: **FROZEN BEFORE ANY V13 OUTPUT**

One design correction is made before execution.

The protocol's candidate K grid `2..10` would exclude the known `KN=16` latent-state count in the mandatory V12 Stage-A positive calibration. That would make the method-qualification test unnecessarily incapable of recovering the generating state cardinality.

Therefore the binding K grid is expanded, before any V13 result is exposed, to:

`K = 2,3,4,...,16`.

All other protocol elements, K-selection rules, familywise nulls, thresholds, parser, corpus, context representation, and stopping rules remain unchanged. The null repeats the full `2..16` K search, so multiplicity remains controlled.
