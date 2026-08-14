# ASC Mechanism × Cipher Order v0.1 — preregistration

Frozen before Phase-6 and Phase-7 adjudications were visible.

## Question

Does the K2 origin-state operation occur before or after the frozen `SWITCH_LINE` cipher transformation?

## Locked contexts

Ordering is tested independently in four predeclared tau=3 state contexts: fixed-continuous, fixed-line-reset, geometric-continuous, and geometric-line-reset. No Phase-6/7 winning context is selected.

For every document/replicate, a single cipher plan is generated from the untouched original plaintext with the existing `P5-plan` seed. That exact plan is reused in both ordering arms.

- POST: plaintext → frozen cipher → K2 state operation.
- PRE: plaintext → K2 state operation → same frozen cipher plan.

Thus only operator order changes.

## Primary statistic

Within each locked context, `Delta_order = robust_d3(PRE) - robust_d3(POST)`. Positive values favour cipher→state. Report median paired-document differences with 10,000-resample 95% bootstrap intervals. Materiality margin: ±0.04 d3.

## Replication guard

Each Phase-8 POST median must reproduce the corresponding Phase-7 arm median to absolute tolerance `1e-12`. Expected values are imported from the immutable Phase-7 final artifact after it completes; they cannot be selected or altered.

## Adjudication

In order: Phase-7 endpoint replication failure; POST required all four; PRE required all four; equivalence all four; robust POST majority; robust PRE majority; same directional median in all four; context-dependent.

The four legacy Q statistics are used only for mechanistic sequencing. Final Voynich confirmation is governed exclusively by the already-sealed Phase-9 consequence panel.
