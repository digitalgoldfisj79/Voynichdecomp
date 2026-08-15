# Amendment 002 — target bridge operationalization

Date: 2026-08-15
Status: pre-inference, pre-Voynich

The original protocol says that, after both external corpora qualify, the Voynich residual must lie within the “broad external operator distribution” on at least two of the H0/H1/delta diagnostics. This amendment fixes that phrase operationally before any external effect sizes or Voynich scores are computed.

For each corpus and diagnostic m in {H0, H1, H0-H1}:

1. Compute per-document operator shifts `D_m = m(ABBREVIATED) - m(EXPANDED)`.
2. Define the ordinary baseline `B_m` as the metric on the pooled EXPANDED corpus.
3. After the external freeze only, compute the Voynich metric `T_m` and target residual `R_m = T_m - B_m`.
4. The diagnostic is “operator-covered” for that corpus iff `R_m` lies inside the empirical 2.5th–97.5th percentile interval of `D_m`.

Strong mechanism support requires operator coverage on at least two of the three diagnostics in **both** Nuremberg and ORIFLAMMS, in addition to the external qualification gates already preregistered.

If both external corpora qualify directionally but this coverage rule fails, the result is `HISTORICAL_ABBREVIATION_DIRECTION_ONLY` rather than mechanism support.

This rule is intentionally conservative. It uses no target value and no observed external effect size. Representational sensitivity checks may be reported descriptively but cannot rescue a failed primary bridge rule.
