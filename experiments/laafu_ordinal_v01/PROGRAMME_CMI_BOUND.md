# LAAFU raw-token conditional-MI bound — preregistration

Date: 2026-08-21. Written before any CMI result was queried.

## Why this bound exists

The full held-out predictive v0.1 workflow is already preregistered and queued on GitHub Actions. This is an independent, cheaper bounding test on the same question; it does not replace or modify v0.1 and cannot rescue a failed v0.1 primary gate.

## Representation

No PGCS, no slot grammar, no morphological decomposition. Source is the stored Zandbergen–Landini ZL3b transliteration (`voynich_nu_data_ZL`) whose database metadata records source URL `https://www.voynich.nu/data/ZL3b-n.txt` and SHA-256 `bf5b6d4ac1e3a51b1847a9c388318d609020441ccd56984c901c32b09beccafc`.

Paragraph loci only. IVTFF annotations are stripped. Tokens containing uncertain/non-letter coding are excluded rather than resolved. Lines require at least 10 retained tokens.

## Question

Does absolute ordinal position within the first six words of a real line predict the current token's first raw EVA character after conditioning on raw local token context?

LEFT events are W2..W6. RIGHT is the exact reversed analogue. The new claim is a graded line-edge relaxation; a uniform edge-zone difference is not the target here.

## Statistic

Conditional mutual information in bits/event:

`I(first_glyph ; ordinal | raw_context)`

Two hierarchical edge tests are fixed before execution:

- **A / immediate relaxation:** C1 on W2..W6, conditioning on the preceding token's raw first character, last character, and length bucket.
- **B / deeper relaxation bound:** C2 on W3..W6, conditioning on the preceding two tokens' raw first characters, last characters, and length buckets.

The restriction of C2 to W3..W6 is a **pre-run measurement-degeneracy correction**: at W2 a two-token history necessarily contains a boundary sentinel, which identifies W2 itself and makes the conditional-independence test ill-posed. This was caught during the audit after the corpus-count query but before any CMI value was computed. C0 (no local context) is descriptive only.

A claim of sustained multi-token relaxation requires both A and B to reach z>=2. A alone means the effect is not resolved beyond one-token local context.

For RIGHT, 'preceding' means preceding in the reversed line, i.e. approaching inward from the physical right edge.

For PHASE, the response is the current first raw EVA character, the coordinate is relative-position quintile, and context is two raw neighboring token shapes on each side. Only positions with two neighbors on each side are eligible.

## Matched null

200 deterministic within-line cyclic-label permutations. Tokens, token order, all raw contexts, line membership, line length, section, and the complete multiset of position labels are held fixed. Only the mapping between a token/context event and the physical coordinate is broken. This directly calibrates sparse-contingency CMI bias.

For edge tests the cyclic shift is over the eligible ordinal labels (five labels for A, four for B). For PHASE it is over each line's observed sequence of relative-position quintile labels. The deterministic shift for line L and replicate r is generated from an MD5 hash of `(L,r)` with a non-zero shift enforced.

## Gates

Headline for each primary/bound: observed CMI, permutation-null mean, null SD, delta, and z=(obs-null_mean)/SD in one sentence.

If z < 2, reporting begins: **the metric does not resolve this**.

LEFT sustained-relaxation claim requires both LEFT-A and LEFT-B z>=2. RIGHT is evaluated analogously as an independent closure-gradient test. PHASE z>=2 after bidirectional raw-neighbor conditioning supports a whole-line relative coordinate.

C0 cannot rescue a failed A/B result. Section analyses are heterogeneity audits only.

## Audit

Circularity: no outcome-derived grammar. Leakage: null relabels coordinates while holding response/context fixed. Confounds: matched within line, so line-level section/length/folio factors are exact. Control fairness: identical estimator on observed/null. Measurement degeneracy: LEFT, RIGHT, PHASE are separate axes; exact left distance, exact right distance, and line length are never jointly conditioned as predictors. The W2 boundary-sentinel degeneracy for C2 was caught and removed before CMI execution. Representation dependence: first raw EVA character primary; raw token length/edge characters only in conditioning. Decision-rule fragility: A/B hierarchy and C0 diagnostic declared before run. Audit completeness: all 200 replicate values are to be retained in the result output.