# Amendment 004 — Q2 fresh qualification and vectorized full-score optimizer

Date: 2026-08-09
Status: **prospective with respect to all binding qualification evidence and all Voynich H17/C17 scoring.**

The first serial run exposed four K=22 control results before cancellation. Those results are now treated as development-only and are not binding evidence.

A control-only benchmark showed that, for K=36, recomputing the exact 36x36 objective in vectorized NumPy is approximately twice as fast as the algebraically equivalent incremental `delta_score` implementation (20,000 proposal benchmark: ~1.08 s full rescoring versus ~2.18 s incremental on the same HF CPU flavor).

The binding v1.7 run therefore uses:

1. a fresh seed namespace `M19STAv17Q2` for control-span selection, synthetic control generation, and optimizer search;
2. Amendment 003's support-aware control span rule, still exclusively within the frozen UD dev+test pools;
3. the exact same annealing proposal kernel, temperature schedule, step counts, restart counts and deterministic polish, but candidate scores are computed by the frozen full `score_num` function rather than `delta_score`.

Before execution, a deterministic self-test checks full-score equality against the previously validated incremental delta implementation on random legal maps. Any discrepancy >1e-10 aborts.

All six controls at every binding K are rerun from scratch. No result from the cancelled first run is used to qualify the instrument. H17/C17 remain sealed until all Q2 qualification gates pass.

No hypothesis, corpus, representation, vocabulary rule, BnF channel, language panel, split, threshold or confirmation criterion is changed.
