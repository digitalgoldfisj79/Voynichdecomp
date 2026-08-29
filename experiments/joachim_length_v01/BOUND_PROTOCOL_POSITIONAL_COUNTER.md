# JLCD v0.1 — positional counter bound

Status: post-primary bounding test, frozen before execution.

Motivation: WLCP v0.1’s strongest surviving length result was not an external-context effect but a positional one: within one-edge-extension families, the longer form is preferentially line-initial rather than line-final. Joachim’s new claim could explain that result only if e/i additions materially participate in the same positional rule.

Test:

1. Discover exact attested one-unit insertion pairs.
2. Select deterministic disjoint e/i insertion pairs and non-e/i insertion pairs (>=8 combined occurrences, >=2 each variant), with no token reused within each set.
3. On physical line edges only, estimate a family/section-stratified Mantel–Haenszel log odds ratio for the longer member being line-initial rather than line-final.
4. Null: independently swap the two observed edge tokens within every physical line with probability 0.5, 5,000 times. This preserves line composition, word frequencies, word lengths, Currier, section and line clustering.
5. Run full corpus, Currier A and Currier B separately.

If e/i counters are the mechanism behind the previously discovered positional length rule, e/i insertion families should reproduce that rule across the full corpus and both Currier systems. Failure does not rule out every length-dependent role for e/i, but it rules out this proposed explanation for the robust WLCP positional signal.
