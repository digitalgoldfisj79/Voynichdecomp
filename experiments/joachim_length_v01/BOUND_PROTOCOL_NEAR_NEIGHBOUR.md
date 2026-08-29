# JLCD v0.1 — post-result bounding protocol: exact one-unit near neighbours

Status: post-primary bounding test, frozen before seeing these bound outputs.

Reason: the primary JLCD-0 test operationalised Joachim’s proposed counters by removing all `e`/`i` units. That may be too aggressive if only *additional* `e`/`i` are counters. This bound therefore tests the narrower claim directly.

## B1 — exact one-unit e/i insertion families

Construct every attested token pair `(short,long)` for which deleting exactly one EVA unit from `long` yields `short`. Primary target pairs are those where the deleted/inserted unit is exactly `e` or `i`.

Select a deterministic disjoint set of target pairs, greedily by minimum variant frequency then total support, so no token type participates in more than one pair. Require at least 8 combined occurrences and at least 2 occurrences of each variant before matching.

Within these exact pairs, test whether short/long variant predicts immediate external context after conditioning on pair identity × Currier × section × line-position bucket × coarse line-length bin. Use conditional mutual information and 500 within-stratum label permutations.

Repeat in Currier A and B and under raw-character representation where the same literal token pair still differs by one `e`/`i` character.

## B2 — non-e/i insertion controls

Repeat B1 on a deterministic disjoint set of equally defined one-unit insertion pairs whose inserted unit is not `e` or `i`. Compare bias-corrected CMI effects and eligible support. A generic near-neighbour effect is not e/i-counter specificity.

## B3 — boundary steps

Report B1 separately for 3→4 and 4→5 pairs, corresponding to the two possible readings of the claimed 3/4 boundary. This is a bound, not a new threshold search. A boundary claim requires the crossing-step effect to exceed 2 null SD in full data and replicate directionally in A and B where sufficient support exists.

## B4 — power injection

On the exact matched e/i pair strata, preserve the number of long and short variants in every stratum but assign long variants with odds ratios 1.5 and 2.0 toward a deterministic half of external-context categories. Across 100 injections per OR, report the fraction exceeding the original null mean + 2 null SD. If the test has poor recovery at OR=2, a null result cannot strongly constrain a context-dependent mechanism of moderate size.

The primary JLCD result is changed only if this narrower operationalisation contradicts it under the registered replication rules. Any contradiction is placed in the retraction list of the final synthesis.
