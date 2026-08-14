# ASC Dwell Regularity v0.1 — preregistration

Frozen 2026-08-14 before any Phase-6 target score is computed.

## Question

At fixed short integrated state correlation time (`tau_int = 3` or `4`), is the Phase-4/5 gain determined by correlation time alone, or does the renewal/dwell law materially matter?

## Invariants

ReM v2.1 diplomatic MHG; 190 eligible documents; first 2000 clean tokens; W=10 artificial lineation; `SWITCH_LINE`; K=2; exact Phase-4 within-token offset rule; continuous state across line boundaries; ATOMIC/LITERAL with worst-representation robust scoring; same scorer, target, cipher schedule, source and permutation depth as Phase 5. Reset semantics are not changed here.

## Laws

For each tau there are three frozen laws:

- fixed: exact Phase-4 fixed-run endpoint (`RUN3` or `RUN4`);
- semi: iid renewal refresh blocks with an intermediate-variance two-point duration law;
- geometric: exact Phase-4 Markov endpoint (`M4` or `M5`).

The semi laws satisfy `tau_int = E[D^2]/E[D]` exactly: `{2,4}` with probabilities `{2/3,1/3}` gives tau=3; `{3,5}` with probabilities `{5/8,3/8}` gives tau=4. A fair state is redrawn independently at each semi-renewal boundary. The geometric-refresh representation is distributionally equivalent to the corresponding binary Markov endpoint.

## Exact endpoint replication guard

Phase-6 plan, endpoint-state and statistic seeds reuse the exact Phase-5 W10 seed namespaces. Before adjudication, the W10 medians for IDENTITY, OCCURRENCE, fixed tau3/tau4, and geometric tau3/tau4 must reproduce the frozen Phase-5 values to absolute tolerance `1e-12`. Failure gives `P5_W10_ENDPOINTS_NOT_REPLICATED` and stops substantive interpretation.

## Primary endpoint

Per document and tau, robust d3 is the worse of ATOMIC/LITERAL `distance_of_median_Q3` (lower is better). Paired differences are:

- FG = geometric - fixed;
- FS = semi - fixed;
- SG = geometric - semi.

Each is summarized by the paired-document median and a 10,000-resample bootstrap 95% CI. Materiality margin is ±0.04 d3, prospectively inherited from the Phase-4 same-law control envelope.

Adjudication is frozen in this order: endpoint replication failure; tau sufficient/no dwell effect; regularity gradient at both tau; gradient at tau3 only; gradient at tau4 only; non-monotonic dwell-law effect; mixed/unresolved. A monotone gradient at a tau requires FG lower CI > +0.04 and both FS and SG lower CIs > 0.

No best-cell selection, tau interpolation, cipher change, line-width retuning, reset change, or posthoc threshold movement is permitted.

Secondary outputs are E1 pairwise contrasts, tau4-minus-tau3 contrasts within each law, representation-specific d3, Q3 component diagnostics, and full-gate hits.
