# ASC-LINE-SCALE-v0.1 preregistration

Protocol SHA-256: `74aaeffd009fbcebaa57327c2867ee15bf501385cdb822c9af9528a98c73a96a`

Parent confirmatory result: `ASC-ORIGIN-MEMORY-v0.1`, frozen adjudication
`INTERMEDIATE_PERSISTENCE_BOTH`.

## Question

Does the Phase-4 short-memory effect occupy an absolute token correlation scale, or does
its preferred scale move when the artificial `SWITCH_LINE` grouping changes?

## Frozen manipulation

Only artificial source-token line width changes: `6, 8, 10, 12, 16, 20`.
For every width the first 2,000 source tokens are retained. The state process is
continuous across line boundaries and is *not* reset by lineation.

Everything else remains frozen: ReM v2.1 diplomatic MHG, 190 eligible documents,
`SWITCH_LINE`, K=2 origin states, half-word K2 offset rule, ATOMIC/LITERAL with
worst-representation robustness, 20 mechanism replicates, 100 permutation replicates,
the Phase-3 scorer/mechanism payload, and target vector
`[1.1642, 1.1039, 1.0257, 1.0182]`.

The exact Phase-4 `persistence_operators.py` is reused unchanged (Git blob
`d6ad9687c7bd76226d37f53324a064909375cb66`).

## Primary contrast

For each document and width:

- fixed short band = RUN2/RUN3/RUN4;
- fixed long band = RUN6/RUN8/RUN12;
- Markov short band = M3/M4/M5;
- Markov long band = M8/M12.

`C(W) = mean(long robust d3) - mean(short robust d3)`.

Positive `C(W)` means the absolute short band is better. The estimator at each width is
the paired-document median with a frozen 10,000-resample bootstrap 95% CI.

For each document, regress `C(W)` on `log2(W/10)` over all six widths. The primary
line-interaction statistic is the median document slope with the same bootstrap rule.

A slope magnitude of 0.04 d3 per doubling is the frozen materiality threshold. It was
chosen *before Phase-5 target scoring* from the Phase-4 same-law control: the paired
OCCURRENCE_K2 vs MARKOV_M2 median-difference bootstrap upper bound was 0.0361, rounded
upward.

## Adjudication

1. If the W=10 replication guard fails -> `P4_CURVE_NOT_REPLICATED_W10`.
2. If both families have lower CI `C(W)>0` at every width and both slope CIs are wholly
   inside [-0.04,+0.04] -> `ABSOLUTE_SHORT_SCALE_ROBUST`.
3. If both families retain lower CI `C(W)>0` at every width but slope equivalence fails
   for at least one family -> `SHORT_SCALE_ROBUST_WITH_LINE_INTERACTION`.
4. Otherwise, if both slope CIs lie wholly below -0.04 ->
   `LINE_RELATIVE_SHIFT_SUPPORTED`.
5. Else -> `MIXED_OR_UNRESOLVED`.

The W=10 replication guard requires OCCURRENCE_K2, FIXED_RUN4_K2 and MARKOV_M4_K2 each
to beat IDENTITY in median robust d3 and on at least 60% of documents.

## Secondary diagnostics

The exact discrete `tau_int` minimum is reported for every width and family, but is not
used as the primary test. Fixed candidates have `tau_int=L`; Markov candidates have
`tau_int=expected_run-1`. Representation-specific curves, Q3 component errors, E1
short-vs-long contrasts, and full-gate hits are descriptive/secondary.

No post-hoc width interpolation, new persistence values, new state alphabet, target-side
phase proxy, or best-cell retuning is permitted in v0.1.

## Execution guard

The workflow file is added only after this protocol, code and target-free QA are committed.
The freeze job verifies this exact SHA and the unchanged Phase-4 state operator before any
scorer payload is restored. All scoring jobs depend on successful freeze.
