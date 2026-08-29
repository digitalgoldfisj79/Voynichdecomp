# JLCD v0.1 — post-result matched specificity contrast

Status: post-primary/post-neighbour bound; frozen before this contrast is executed.

The exact-neighbour bound found no resolved e/i insertion effect while non-e/i insertions showed a larger effect. Because those pair sets were not support-matched, this protocol tests the specificity contrast directly.

1. Start from attested one-unit insertion pairs with >=8 combined occurrences and >=2 occurrences of each variant.
2. Build the deterministic disjoint e/i target set as in the near-neighbour bound.
3. For each target pair, greedily select one non-e/i pair with the same short-token EVA length and closest `(log2 short_count, log2 long_count, log2 total_count)`, without reusing any token type in either target or control set. Skip targets with no eligible control.
4. Apply the same pair × Currier × section × line-position × line-length matching and external-context CMI statistic to both sets.
5. Run 2,000 synchronized within-stratum permutations. The contrast statistic is the bias-corrected target effect minus bias-corrected control effect. Report its null SD and standardized effect.
6. Repeat under EVA-unit and raw-character context/length representation.

Joachim-specificity support requires a positive contrast >=2 null SD in both representations. A negative contrast >=2 null SD is evidence against e/i being unusually context-disruptive among equally supported one-unit variants, but does not rule out every possible cipher role for e/i.
