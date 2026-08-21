# LAAFU raw-token conditional-MI bound — preregistration

Date: 2026-08-21. Written before any CMI result was queried.

## Why this bound exists

The full held-out predictive v0.1 workflow is already preregistered and queued on GitHub Actions. This is an independent, cheaper bounding test on the same question; it does not replace or modify v0.1 and cannot rescue a failed v0.1 primary gate.

## Representation

No PGCS, no slot grammar, no morphological decomposition. Source is the stored Zandbergen–Landini ZL3b transliteration (`voynich_nu_data_ZL`) whose database metadata records source URL `https://www.voynich.nu/data/ZL3b-n.txt` and SHA-256 `bf5b6d4ac1e3a51b1847a9c388318d609020441ccd56984c901c32b09beccafc`.

Paragraph loci only. IVTFF annotations are stripped. Tokens containing uncertain/non-letter coding are excluded rather than resolved. Lines require at least 10 retained tokens.

## Question

Does absolute ordinal position within the first six words of a real line predict the current token's first raw EVA character after conditioning on raw local token context?

Primary LEFT events are W2..W6. RIGHT is the exact reversed analogue. The new claim is a graded line-edge relaxation; a uniform edge-zone difference is not the target here.

## Statistic

Conditional mutual information in bits/event:

`I(first_glyph ; ordinal | raw_context)`

Primary raw context C2 is the preceding two tokens represented only by transcription-level observables: first character, last character, and length bucket. C1 (one preceding token) and C0 (no context) are sensitivity diagnostics, not alternative primaries.

For RIGHT, 'preceding' means preceding in the reversed line, i.e. approaching inward from the physical right edge.

For PHASE, the response is the current first raw EVA character, the coordinate is relative-position quintile, and context is two raw neighboring token shapes on each side. Only positions with two neighbors on each side are eligible.

## Matched null

200 deterministic within-line cyclic-label permutations. Tokens, token order, all raw contexts, line membership, line length, section, and the complete multiset of position labels are held fixed. Only the mapping between a token/context event and the physical coordinate is broken. This directly calibrates sparse-contingency CMI bias.

The deterministic shift for line L and replicate r is generated from an MD5 hash of `(L,r)` modulo the number of coordinate labels, with a non-zero shift enforced.

## Gates

Headline for each axis: observed CMI, permutation-null mean, null SD, delta, and z=(obs-null_mean)/SD in one sentence.

If z < 2, reporting begins: **the metric does not resolve this**.

LEFT C2 z>=2 supports a graded left-edge coordinate that is not eliminated by two-token raw-shape context. RIGHT C2 is an independent closure test. PHASE z>=2 after bidirectional raw-neighbor conditioning supports a whole-line relative coordinate.

C0/C1 cannot rescue a failed C2 primary result. Section analyses are heterogeneity audits only.

## Audit

Circularity: no outcome-derived grammar. Leakage: null relabels coordinates while holding response/context fixed. Confounds: matched within line, so line-level section/length/folio factors are exact. Control fairness: identical estimator on observed/null. Measurement degeneracy: LEFT, RIGHT, PHASE are separate axes; exact left distance, exact right distance, and line length are never jointly conditioned as predictors. Representation dependence: first raw EVA character primary; raw token length/edge characters only in conditioning. Decision-rule fragility: C0/C1 diagnostics declared before run. Audit completeness: all 200 replicate values are to be retained in the result output.