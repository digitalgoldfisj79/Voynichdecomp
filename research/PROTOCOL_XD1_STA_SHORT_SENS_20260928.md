# XD1 STA short-word decision-rule sensitivity

**Protocol ID:** XD1-STA-SHORT-SENS-20260928
**Frozen:** after the primary STA/RF result. This is explicitly POST-OUTCOME and cannot upgrade the claim; it can only downgrade it if the alternative word-boundary convention reverses the result.

## Reason
The source-only preflight established two legitimate RF1b token-boundary conventions:
- long-word: <-> is not a boundary (37,087 words);
- short-word: <-> is a boundary (37,848 words).

Because the primary test used the long-word convention, the short-word convention is a direct decision-rule-fragility check.

## Scope and statistic
Same RF1b STA1 source, canonical folios and P loci as XD1-STA-RF-20260928.
Split words on either '.' or '<->'. No EVA mapping.
For lag 1 and lag 2 use the same within-physical-line exact-token multiset permutation null, 2,000 permutations, analytic-null cross-check, and 2,000 line-bootstrap replicates.

## Downgrade rule
The primary representation result is decision-rule-fragile if either:
- lag 2 is not positive with effect/null-SD >=2; or
- lag 1 becomes significantly suppressive (signed effect/null-SD <= -2).

Passing this sensitivity does not create a new independent replication.
