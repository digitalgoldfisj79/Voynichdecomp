# LAAFU raw-token conditional-MI bound — preregistration / running audit

Date: 2026-08-21.

## RETRACTED FINDINGS

**2026-08-21 — RETRACTED / VOID: first LEFT-A CMI run used an invalid null.** It produced observed 0.477507 bits/event versus null 0.536664, SD 0.006273, z=-9.43. Do not interpret or cite this as evidence. The within-line cyclic *coordinate-label* permutation changed the joint distribution of raw context C and ordinal K, so it was not a valid conditional-independence null for `I(G;K|C)`. This control-fairness failure was identified immediately on inspecting the direction of the null and before any corrected CMI result was run. The corrected null below preserves both C–K and G–C exactly.

## Why this bound exists

The full held-out predictive v0.1 workflow is already preregistered and queued on GitHub Actions. This is an independent, cheaper bounding test on the same question; it does not replace or modify v0.1 and cannot rescue a failed v0.1 primary gate.

## Representation

No PGCS, no slot grammar, no morphological decomposition. Source is the stored Zandbergen–Landini ZL3b transliteration (`voynich_nu_data_ZL`) whose database metadata records source URL `https://www.voynich.nu/data/ZL3b-n.txt` and SHA-256 `bf5b6d4ac1e3a51b1847a9c388318d609020441ccd56984c901c32b09beccafc`.

Paragraph loci only. IVTFF annotations are stripped. Tokens containing uncertain/non-letter coding are excluded rather than resolved. Lines require at least 10 retained tokens. The audit count, obtained before the corrected CMI run, is 1,431 eligible lines on 136 folios, containing 16,199 retained tokens; median eligible-line length is 11.

## Question

Does absolute ordinal position within the first six words of a real line predict the current token's first raw EVA character after conditioning on raw local token context?

LEFT events are W2..W6. RIGHT is the exact reversed analogue. The new claim is a graded line-edge relaxation; a uniform edge-zone difference is not the target here.

## Statistic

Conditional mutual information in bits/event:

`I(first_glyph ; ordinal | raw_context)`

Two hierarchical edge tests were fixed before corrected execution:

- **A / immediate relaxation:** C1 on W2..W6, conditioning on the preceding token's raw first character, last character, and length bucket.
- **B / deeper relaxation bound:** C2 on W3..W6, conditioning on the preceding two tokens' raw first characters, last characters, and length buckets.

Section, exact retained line length, and paragraph-start status are also included in the conditioning stratum. They are nuisance controls, not Voynich grammar.

The restriction of C2 to W3..W6 is a pre-run measurement-degeneracy correction: at W2 a two-token history necessarily contains a boundary sentinel, which identifies W2 itself and makes the conditional-independence test ill-posed. This was caught before any CMI value was computed. C0 (no local context) is descriptive only.

A claim of sustained multi-token relaxation requires both A and B to reach z>=2. A alone means the effect is not resolved beyond one-token local context.

For RIGHT, 'preceding' means preceding in the reversed line, i.e. approaching inward from the physical right edge.

For PHASE, the response is the current first raw EVA character, the coordinate is relative-position quintile, and context is two raw neighboring token shapes on each side. Only positions with two neighbors on each side are eligible.

## CORRECTED matched null

200 deterministic **conditional-response permutations**. Within each complete conditioning stratum C, the observed first-glyph responses G are permuted among events while the ordinal/phase coordinate K stays fixed. Thus the null preserves exactly:

- the C–K joint distribution;
- the G–C joint distribution and every context-stratum response multiset;
- section, exact retained line length and paragraph-start composition through their inclusion in C;
- the estimator, sample size and sparse-context structure.

Only the residual G–K association conditional on C is broken. Permutations are deterministic from `(event_id, replicate)` hashes.

This supersedes the retracted coordinate-label null above. The change is methodological, not outcome-selected: the first null failed the defining invariance requirement of a conditional-independence test.

## Gates

Headline for each primary/bound: observed CMI, permutation-null mean, null SD, delta, and z=(obs-null_mean)/SD in one sentence.

If z < 2, reporting begins: **the metric does not resolve this**.

LEFT sustained-relaxation claim requires both LEFT-A and LEFT-B z>=2. RIGHT is evaluated analogously as an independent closure-gradient test. PHASE z>=2 after bidirectional raw-neighbor conditioning supports a whole-line relative coordinate.

C0 cannot rescue a failed A/B result. Section analyses are heterogeneity audits only.

## Audit

Circularity: no outcome-derived grammar. Leakage: the corrected null holds C–K and G–C fixed and permutes only G inside C. Confounds: section, exact line length and paragraph-start status are part of C; every eligible line contributes the same ordinal set in edge tests. Control fairness: identical CMI estimator and identical context strata in observed/null. Measurement degeneracy: LEFT, RIGHT, PHASE are separate axes; the W2 boundary-sentinel problem is excluded from C2. Representation dependence: first raw EVA character primary; raw token length/edge characters only in conditioning. Decision-rule fragility: A/B hierarchy and C0 diagnostic fixed before the corrected run. Audit completeness: all 200 corrected replicate values are to be retained in the result output.