# Phase 4 post-hoc diagnostic decomposition

Status: **post-hoc diagnostic only**. These calculations were performed after the frozen
ASC-ORIGIN-MEMORY-v0.1 result was known. They motivate Phase 5 but are not counted as
confirmatory evidence.

## 1. Representation-specific d3 curves

Median `distance_of_median_Q3` by representation:

| arm | ATOMIC | LITERAL |
|---|---:|---:|
| IDENTITY | 0.9044 | 0.7879 |
| OCCURRENCE_K2 | 0.7864 | 0.6849 |
| FIXED_RUN2_K2 | 0.5758 | 0.4091 |
| FIXED_RUN3_K2 | 0.5542 | 0.4133 |
| FIXED_RUN4_K2 | 0.5889 | 0.4363 |
| FIXED_RUN6_K2 | 0.6662 | 0.5197 |
| FIXED_RUN8_K2 | 0.7178 | 0.5708 |
| FIXED_RUN12_K2 | 0.8142 | 0.6749 |
| MARKOV_M2_K2 | 0.7631 | 0.6666 |
| MARKOV_M3_K2 | 0.6326 | 0.4979 |
| MARKOV_M4_K2 | 0.6181 | 0.4927 |
| MARKOV_M5_K2 | 0.6278 | 0.4760 |
| MARKOV_M8_K2 | 0.6779 | 0.5338 |
| MARKOV_M12_K2 | 0.7229 | 0.5978 |

The U-shaped short-persistence effect is therefore present in both representations and is
not created solely by the worst-case robust aggregation.

## 2. Q3 component decomposition

Median absolute log-error versus the frozen target, shown for the main fixed sequence:

| arm | ATOMIC Q1 | Q2 | Q3 | LITERAL Q1 | Q2 | Q3 |
|---|---:|---:|---:|---:|---:|---:|
| IDENTITY | .645 | .473 | .435 | .561 | .387 | .333 |
| OCCURRENCE_K2 | .577 | .400 | .345 | .519 | .333 | .294 |
| FIXED_RUN2_K2 | .397 | .284 | .270 | .286 | .213 | .191 |
| FIXED_RUN3_K2 | .393 | .271 | .258 | .286 | .196 | .197 |
| FIXED_RUN4_K2 | .409 | .297 | .275 | .301 | .218 | .204 |
| FIXED_RUN6_K2 | .456 | .358 | .325 | .337 | .260 | .238 |
| FIXED_RUN8_K2 | .498 | .378 | .363 | .398 | .280 | .258 |
| FIXED_RUN12_K2 | .557 | .411 | .401 | .476 | .330 | .305 |

The short-memory gain is distributed across ED1_N0, ED1_N1 and ED1_N3 rather than being
carried by a single Q3 coordinate.

## 3. Same-law control

`OCCURRENCE_K2` and `MARKOV_M2_K2` are independent constructions of the same fair
independent binary-state law. On the robust paired-document d3 difference
`OCCURRENCE_K2 - MARKOV_M2_K2`:

- n = 189;
- median = 0.0120;
- mean = 0.0238;
- MARKOV_M2 is lower on 55.0% of documents;
- 10,000-resample bootstrap 95% CI for the paired median = [-0.0056, 0.0361].

Thus the aggregate median difference between those arms is not a stable law difference.
For Phase 5, 0.04 d3 is frozen as a conservative materiality envelope for an interaction
slope per doubling of line width.

## 4. Correlation-time reparameterisation

For fixed blocks, the integrated autocorrelation time is `tau_int=L`. For the symmetric
Markov process used here, `tau_int=expected_run-1`.

At the Phase-4 reference width W=10:

- fixed-family median robust d3 minimum: RUN3 -> `tau_int=3`;
- Markov-family median robust d3 minimum: M4 -> `tau_int=3`.

This alignment was *not* a preregistered Phase-4 endpoint. Phase 5 therefore treats it as
a hypothesis to be challenged, not as an established result.

## 5. Immediate confound

All Phase-4 scoring used artificial groups of 10 source tokens under `SWITCH_LINE`.
Therefore an apparent 3-4-token memory scale might be an interaction with the imposed
line/chunk scale. Phase 5 changes only that scale while holding source, scorer, K=2 state
operator, cipher schedule, target, representations and Monte-Carlo settings fixed.
