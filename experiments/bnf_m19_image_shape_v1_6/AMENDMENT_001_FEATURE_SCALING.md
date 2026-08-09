# Amendment 001 — topology feature scaling

Date: 2026-08-09
Timing: prospective; created before the first v1.6 shape-feasibility run and before any v1.6 visual score was observed.

Clarification to PROTOCOL.md Stage S0 step 4:

- The topology-only representation **T** is standardized featurewise with `StandardScaler` fit on ShapeTrain only, then L2-normalized rowwise before clustering.
- The standardized topology block used in HT/RT/HRT is the same ShapeTrain-fitted transformation.
- PCA and all other scalers remain fit on ShapeTrain only and applied unchanged to ShapeTest.

No gate, split, feature definition, K value, seed, sample size or selection rule is changed.
