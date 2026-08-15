# STA boundary-return discriminator v0.1 — implementation freeze

Frozen before target model execution. This file supplements `PREREG_20260815.md`; it does not change its hypotheses or gates.

## Emission backoff/support

All empirical emission counters must have at least **30 token observations and 5 distinct token types** to be used. Otherwise the implementation backs off deterministically.

B0 backoff order:
1. `(section, length_bucket, edge_coordinate)`;
2. `(section, edge_coordinate)`;
3. `(section, length_bucket)`;
4. `section`;
5. global.

B1 adds `(section, exact_n, edge_coordinate)` ahead of B0.

B2 adds, in order:
1. `(Z, section, exact_n, edge_coordinate)`;
2. `(Z, section, length_bucket, edge_coordinate)`;
3. `(Z, edge_coordinate)`;
4. B1 backoff chain.

Template-prior `P(Z|...)` requires at least 10 training lines at a conditioning level. Prior backoff is `(section,length_bucket)` -> `section` -> global.

## Template clustering

Primary `K=8` uses deterministic categorical k-modes with Hamming distance.

Initialization:
- first prototype = most frequent boundary-feature vector, lexicographic tie-break;
- remaining prototypes = farthest-first unique vector from existing prototypes, lexicographic tie-break.

Update:
- assign to nearest prototype, lowest-index tie-break;
- each categorical coordinate updates to its modal value, lexicographic tie-break;
- an empty cluster retains its previous prototype;
- stop on unchanged assignments or after 30 iterations.

No random initialization is permitted.

## Return parameter fitting

For B3 and CG, q is selected on the fixed grid `0.0000, 0.0005, ..., 0.2500` by maximum training conditional log-likelihood; lowest-q tie-break.

The B2 marginal probability used for q fitting and held-out prediction marginalizes over the training-only template prior. Emission probabilities use the same frozen backoff chain. Seen-vocabulary probabilities receive Jeffreys/KT smoothing `(count+0.5)/(N+0.5*V)` over the fold's training vocabulary. Unseen held-out target tokens are excluded from the predictive gate exactly as preregistered.

## Randomness

Base seed: `20260815`.
- Folio folds are inherited SHA-256 folds.
- OOF generation: deterministic seed family keyed by replicate, fold and model.
- Folio bootstrap: 2,000 deterministic resamples using seed `20260815 + 700000`.
- CN1 controls: replicate-specific deterministic N1 shuffles.

## Real-anchor validation before target adjudication

The implementation must reproduce the inherited STA-family exact-lag-2 geometry within these non-decision validation windows before target models are scored:
- E2 N0 in `[1.16,1.20]`;
- E2 N1 in `[1.05,1.09]`;
- E2 N3 in `[1.02,1.07]`.

Failure is `IMPLEMENTATION_VALIDATION_FAIL`; model output cannot be interpreted.

## Sensitivity discipline

K=4 and K=12 are reported only after the K=8 primary verdict is computed. They may diagnose brittleness but may not promote or rescue a primary model.