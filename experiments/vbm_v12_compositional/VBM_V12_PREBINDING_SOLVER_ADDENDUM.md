# VBM v12 — pre-binding solver addendum

Date: 2026-09-02
Status: **FROZEN BEFORE ANY BINDING V12 SCIENTIFIC OUTPUT**
Parent protocol: `VBM_V12_COMPOSITIONAL_TRANSDUCER_PROTOCOL.md`

## Why this addendum exists

A software-only smoke study performed before any binding V12 replicate was exposed showed that random-start likelihood climbing is a poor instrument for M12 for the same conceptual reason identified in V10: a higher source likelihood need not move a candidate key toward the generating key.

The binding V12 solver is therefore changed **before scientific output** from blind random-start likelihood optimisation to a deterministic context-partition and algebraic-projection solver. The model, source families, corpus sizes, adversaries, recovery metrics, gates, and interpretation firewall are unchanged.

This addendum supersedes only the `Solver` section of the parent protocol.

## Binding solver

The solver receives exactly the information listed in the parent protocol and no true-key information.

### 1. FIT transition counts

Aggregate FIT counts into:

- `C_NB[tN,tB]`: occurrences of nucleus surface type `tN` immediately followed by bridge surface type `tB`;
- `C_BN[tB,tN]`: occurrences of bridge surface type `tB` immediately followed by nucleus surface type `tN`.

HOLDOUT is never used in clustering, alignment, projection, restart selection, or tuning.

### 2. Surface context features

With additive smoothing 0.5:

For each nucleus surface type concatenate:

- its row-normalised distribution over following bridge surface types from `C_NB`;
- its column-normalised distribution over preceding bridge surface types from `C_BN`.

For each bridge surface type concatenate:

- its column-normalised distribution over preceding nucleus surface types from `C_NB`;
- its row-normalised distribution over following nucleus surface types from `C_BN`.

Apply elementwise square root to the normalised probabilities (Hellinger embedding).

### 3. Anonymous state partitions

Cluster nucleus surface features into exactly `KN` clusters and bridge surface features into exactly `KB` clusters using k-means++.

Binding restarts:

- Stage A: 16 deterministic KMeans random states `0..15`, each with `n_init=32`;
- Stage B: 24 deterministic KMeans random states `0..23`, each with `n_init=32`.

Each paired nucleus/bridge clustering restart is carried forward independently through alignment and algebraic projection.

### 4. Align anonymous clusters to the supplied source

No truth map is used.

From each anonymous clustering construct empirical cluster-level transition matrices `Q(Bcluster|Ncluster)` and `Q(Ncluster|Bcluster)` with smoothing 0.5.

For each nucleus cluster form a permutation-invariant signature by concatenating:

- sorted outgoing probabilities across bridge clusters;
- sorted incoming probabilities across bridge clusters.

Construct the analogous invariant signature for each true source nucleus state from the supplied `P(B|N)` and `P(N|B)` matrices. Align anonymous nucleus clusters to source nucleus states with minimum-squared-distance Hungarian assignment.

Do the same independently for bridge clusters using sorted outgoing `P(N|B)` and incoming `P(B|N)` signatures.

This alignment uses the known source matrices, not generating surface keys.

### 5. Project nucleus partition onto the M12 e-operator

Given aligned estimated nucleus-state labels for visible types `(s,m)`:

- initialise `base[s]` from the aligned label of `(s,0)`;
- construct a `KN x KN` count matrix of observed aligned transitions from label `(s,m)` to label `(s,m+1)` over every skeleton and adjacent e-level;
- infer the global permutation `pi` by maximum-weight bipartite assignment (Hungarian algorithm);
- induce the complete M12 nucleus map `pi^m(base[s])`.

No direct e-count-to-length interpretation is introduced.

### 6. Project bridge partition onto additive visible halves

Given aligned estimated bridge-state labels `Y[R,L]`:

- fix the harmless gauge with `u[0]=0`;
- initialise `v[L]=Y[0,L]`;
- alternately conditionally maximise each `u[R]` and each `v[L]` over `0..KB-1` for exact agreement with `Y`, for at most 50 sweeps or until unchanged;
- induce the complete bridge map `(u[R]+v[L]) mod KB`.

### 7. Restart selection

For every clustering restart, compute FIT **partition reconstruction accuracy without truth**:

- fraction of FIT nucleus occurrences whose aligned anonymous cluster label equals the induced M12 nucleus state;
- fraction of FIT bridge occurrences whose aligned anonymous cluster label equals the induced additive bridge state;
- occurrence-weighted joint mean of those two fractions.

Select the restart with the highest joint reconstruction score; break ties by higher FIT source log likelihood, then lower restart index.

Generating truth is never used for restart selection.

### 8. Source likelihood

Source log likelihood is an evaluation and tie-break statistic only. No post-projection likelihood climbing is permitted in the binding solver.

This explicitly avoids recreating V10's demonstrated failure mode in which likelihood improvement can move away from the generating key.

## Implementation smoke gate

Before binding Stage A output, one non-binding miniature synthetic corpus with dimensions strictly smaller than Stage A must verify:

- all generated latent states have at least one compatible surface type;
- context matrices have expected shapes;
- anonymous clustering returns exactly `KN` and `KB` nonempty clusters;
- Hungarian alignments are bijections;
- projected `pi` is a permutation;
- bridge gauge projection is deterministic;
- no HOLDOUT counts enter FIT features or restart selection.

The smoke is a software qualification only and may not be reported as V12 evidence.

## Everything else remains frozen

The following parent-protocol items are unchanged:

- M12 algebra;
- Stage A and Stage B dimensions;
- source families and replicate counts;
- adversarial families;
- HOLDOUT recovery metrics;
- all numeric gates;
- Stage B opening rule;
- Voynich/plaintext firewall.
