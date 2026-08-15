# Target Compatibility Amendment 001 — collinearity and family-balance preflight

Date: 2026-08-15
Status: **pre-target amendment**. No target entropy values have been computed in this stage.

A hostile review of the frozen target protocol identified three design defects before target execution:

1. `MI1 = H0 - H1` is algebraically determined by H0 and H1, so including all three in the distance vector would double-count the same information.
2. `H1_norm` is also derived from H1 and alphabet size and should not be treated as an independent dimension.
3. Pooling all external windows would overweight source families with more windows, especially the larger Latin corpora, and a pooled plaintext-null percentile would therefore be pseudo-replicated rather than family-balanced.

The following corrections supersede the relevant target-protocol clauses without changing any historical mechanism or target representation:

## Independent primary distance vector
Primary distance uses exactly **H0 and H1**.

`MI1`, `H1_norm`, `H2`, and `K` are secondary diagnostics only and cannot change the primary verdict.

An individual mechanism is externally qualified when both H0 and H1 have finite nonzero robust scale.

## Family-balanced external model
For each mechanism and metric:
- first compute the median separately within each source family;
- model centre is the median of the source-family medians;
- model scale is `1.4826 * MAD` of the source-family medians; if zero, use the standard deviation of the source-family medians; if still zero, the mechanism is unqualified.

LOSO fitting uses the same rule on the remaining five source-family medians. Thus each external source family casts one calibration vote regardless of corpus size.

## Statistic-matched plaintext null
The target statistic is the **median window-level group advantage**. The external plaintext null is therefore calibrated with the same statistic:
- for each held-out source family, calculate the median `Delta_group` across that family's held-out identity windows;
- the group null threshold is the **maximum of these six held-out-family medians**.

This replaces the pooled 95th-percentile rule and prevents large source families from dominating the null.

Positive-control qualification remains family-balanced: median generated-cipher advantage must be positive overall and positive in at least 5/6 held-out source-family medians.

These repairs are driven entirely by algebra and experimental-design principles. They introduce no target values, target-selected threshold, or target-dependent mechanism choice.
