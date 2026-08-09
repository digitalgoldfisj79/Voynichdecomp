# Amendment 002 — control-span seed and optimizer self-test

Date: 2026-08-09
Status: **prospective; no language score had been generated when this amendment was committed.**

Two implementation corrections are made without changing any model, data, threshold, split, representation, language panel, or qualification criterion:

1. The control-span selector will use the frozen `M19STAv17` SHA-256 seed namespace directly instead of inheriting the older v0.7 helper's seed namespace.
2. Before source/model scoring, the runner will perform deterministic random legal-map checks that compare the incremental annealer `delta_score` against a full `score_num` recomputation. Any discrepancy >1e-10 aborts the programme before language scoring.

The executed entry point is `run_v17_amended.py`; it imports the frozen `run_v17.py` implementation and applies only these two changes.
