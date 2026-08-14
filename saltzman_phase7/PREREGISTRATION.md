# ASC Reset Semantics v0.1 — preregistration

Frozen before Phase-6 adjudication is visible.

## Question

At fixed `tau_int=3`, does the K2 origin-state process need to persist across the 10-token artificial line boundary used by `SWITCH_LINE`?

## Design

All Phase-5/6 invariants remain fixed: canonical ReM v2.1 diplomatic MHG, 190 eligible documents, first 2000 tokens, W=10, `SWITCH_LINE`, K2 half-rotation, ATOMIC/LITERAL robust worst-case scoring, 20 document replicates and 100 permutation replicates.

Two dwell-law strata are locked in parallel, rather than selected from Phase 6:

- fixed tau=3: continuous `FIXED_RUN3` versus an independently restarted `FIXED_RUN3` process at each line;
- geometric tau=3: continuous `MARKOV_M4` versus an independently restarted `MARKOV_M4` process at each line.

The intervention is reset semantics within each stratum. No Phase-6 best-cell choice enters Phase 7.

## Primary statistic

For each document and stratum,

`Delta_reset = robust_d3(line_reset) - robust_d3(continuous)`.

Positive values favour cross-line continuity. Report the median paired-document difference with a 10,000-resample 95% bootstrap CI. The inherited materiality margin is ±0.04 d3.

## Replication guard

Continuous fixed and geometric endpoints must reproduce their frozen W10 medians exactly to absolute tolerance `1e-12`. If the Phase-6 endpoint replication guard fails, Phase 7 is not launched.

## Adjudication

In order: endpoint replication failure; continuity required in both strata; line reset better in both; reset equivalence in both; materially opposite dwell-dependent reset; consistent direction without robust materiality; mixed/unresolved.

No post-hoc boundary scale, tau, dwell-law parameter, representation, scorer, language or cipher tuning is permitted.
