# VBM v12 — implementation specification

Date: 2026-09-02
Status: **FROZEN BEFORE BINDING V12 OUTPUT**

This file fixes execution details not fully specified in the parent protocol and pre-binding solver addendum.

## Deterministic namespace

All pseudo-random streams use SHA256-derived seeds under namespace:

`VBMV12COMPOSITIONAL20260902`

The seed tag includes stage, source family, replicate and mode as applicable.

## Synthetic key construction

For positive M12 replicates:

- `base[s]` is balanced over `KN` by round-robin assignment then deterministic shuffle;
- `pi` is a deterministic random permutation of `KN`;
- `u[R]` and `v[L]` are balanced round-robin assignments over `KB` then deterministically shuffled;
- induced nucleus and bridge maps are exactly the functions frozen in the parent protocol.

For adversarial controls, broken full maps are balanced round-robin assignments over the relevant latent alphabet followed by deterministic shuffle. The unbroken component remains the positive M12 component.

## Surface emission propensities

Each replicate receives fixed positive surface weights drawn as `exp(N(0,0.7))`. The same weight vector is reused across positive and adversarial modes for that replicate. Conditional on a latent state, compatible surface types are sampled proportional to these public weights. These weights are generation-only nuisance parameters and are not used by the solver.

## Lines

Each line contains a deterministic random integer from 5 through 15 inclusive nucleus events. Every line begins with a nucleus state and alternates `N,B,N,...,N`. The first 80% of generated lines are FIT and the final 20% are untouched HOLDOUT.

## Context features

The exact feature construction is the solver addendum's two-sided Hellinger embedding with pseudocount 0.5. KMeans uses `k-means++`, `n_init=32`, with nucleus random state `seed` and bridge random state `1000+seed`.

## Source-state alignment

Anonymous cluster labels are aligned independently for nucleus and bridge states using the permutation-invariant sorted transition signatures in the solver addendum. Squared Euclidean distance and SciPy Hungarian assignment are binding.

## Nucleus projection

After Hungarian recovery of `pi` from the aligned e-level transition count matrix, each skeleton base value is chosen to maximise exact agreement across all available e-levels. Ties are resolved by the numerically smallest latent state.

## Bridge projection

Gauge is fixed with initial `u[0]=0` and `v[L]=Y[0,L]`. Alternating exact coordinate maximisation is run for at most 50 sweeps. Ties choose the numerically smallest latent value. The complete induced bridge map, not raw half labels, is primary.

## Restart selection

For each restart, calculate occurrence-weighted agreement between the aligned anonymous surface-state partition and the algebraically projected M12 map separately for nuclei and bridges. The joint reconstruction score weights by FIT occurrence counts across both factor families.

Select the largest joint reconstruction score. Ties: larger FIT source-transition log likelihood, then lower restart index.

No true generating map participates in selection.

## Likelihood / regret

`HOLD_LM_TRUE` and `HOLD_LM_FIT` are mean log probabilities per **NB or BN transition** under the supplied source transition matrices. Line-initial `P(N0)` is not included in these reported regret metrics. `HOLD_REGRET = HOLD_LM_TRUE - HOLD_LM_FIT`.

Source likelihood is never climbed after algebraic projection.

## Recovery

All primary recovery metrics are occurrence-weighted on HOLDOUT.

Frequent-only metrics include a surface type only if it appeared at least five times in FIT.

`REC_HALF_GAUGE` is the maximum, over every additive gauge shift `c`, of joint unweighted half-component agreement under `u_hat = u_true+c` and `v_hat = v_true-c (mod KB)`.

## Software smoke

The mandatory non-binding smoke uses dimensions strictly smaller than Stage A:

- `KN=8`, `KB=4`;
- 16 skeletons × 3 e-levels;
- 6 right halves × 5 left halves;
- 300 lines, 80/20 split;
- 4 clustering restarts.

The smoke is software qualification only and cannot contribute to any Stage-A or Stage-B gate.
