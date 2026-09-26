# LAAFU ordinal-position test v0.1 — preregistration

Date: 2026-08-21

## RETRACTED / EXCLUDED FRAMING

PGCS is **not** treated as a uniquely identified grammar or ontology in this experiment. Prior work showed that a four-slot factorisation is not uniquely preferred. No PGCS slot, PGCS parse, morphological family, or other hand-built Voynich grammar enters the response, predictors, nulls, or decision rule.

## Prior result being extended

Existing LAAFU work already supports special line boundaries and weak/failed interior effects under several controls. The new Claude/JoJo follow-on suggests a graded W2→W5/6 dependence rather than merely FIRST/MID/LAST. The test below asks whether that graded dependence survives raw-token local-context controls.

## Hypothesis

H0 (transition/geometry): once raw local sequence, section, Currier, paragraph state and line length are controlled, token form is conditionally independent of physical position within the line. Any W2→W6 profile is a transient induced by the boundary plus ordinary local transitions and/or finite-line geometry.

H1 (independent line coordinate): physical distance from a real line edge, or relative phase within the line, adds held-out predictive information about token form after those controls.

## Corpus

Primary source: René Zandbergen / Landini ZL3b-n.txt from voynich.nu, downloaded at run time. Paragraph loci only (`P*` loci); labels/radial/circular registers excluded. Conservative raw EVA token cleaning excludes uncertain/alternate-coded tokens rather than resolving them.

Primary edge corpus: lines with at least 10 retained tokens. W2…W6 are analysed, leaving at least three retained tokens beyond W6. This prevents the LEFT test from mechanically sampling the closing zone. RIGHT is the exact reversed analogue.

## Representation

No word decomposition. Primary response = first raw EVA character of the current token. Diagnostics = last raw EVA character, token-length class, and common raw token identity (top 64 types vs OTHER).

Baseline predictors are raw and non-PGCS: section, Currier, paragraph-start flag, total retained line length, the anchor token, and up to three preceding raw tokens plus their length/edge-character features. RIGHT reverses the line. Relative-PHASE uses two raw neighbours on each side plus both physical edge tokens.

## Leakage / circularity / degeneracy controls

- Evaluation is grouped by quire using held-out GroupKFold.
- Feature hashing is stateless; no feature vocabulary is learned from held-out data.
- LEFT and RIGHT are tested separately. The model never conditions simultaneously on exact line length, exact left distance and exact right distance; that would make position algebraically recoverable.
- PHASE is tested separately and conditions on local context from both directions.
- No outcome-derived Voynich grammar enters the predictor set.

## Matched nulls

### N1 — internal-anchor null (LEFT / RIGHT)

For every eligible line, move the pseudo-edge to a later contiguous six-token window in the **same line**, choosing an anchor that still leaves three tokens after pseudo-W6. This preserves the exact line, token multiset, section, Currier, total line length, and all transitions inside the six-token window. The baseline is rebuilt from that pseudo-edge. This is the primary transition-preserving null.

### N2 — within-line phase permutation (PHASE)

Keep tokens and all baseline contexts fixed; cyclically permute relative-position quintile labels within each line. This preserves each line’s exact phase-label multiset while breaking token↔physical-phase alignment.

Thirty matched-null replicates are preregistered for v0.1.

## Metric

For each response task, fit identical regularised log-loss linear classifiers with and without the coordinate feature. Primary effect size:

`gain = held-out NLL_baseline − held-out NLL_baseline+position` in bits/event.

Positive gain means the coordinate adds predictive information beyond the baseline.

Primary headline is the first-glyph gain. Report observed gain, matched-null mean, null SD, delta, and `z=(observed-null_mean)/null_SD` in the same sentence.

## Decision gates

- If z < 2: lead with **“the metric does not resolve this.”**
- If z ≥ 2 but fewer than 60% of held-out quire groups have positive OOF gain: call it **localized, not corpus-wide**.
- If z ≥ 2 and at least 60% of held-out quire groups are positive: the axis **survives the primary gate**.

LEFT survival with PHASE failure supports a decaying line-start/reset process rather than whole-line planning. Independent RIGHT survival supports a closure process. PHASE survival after bidirectional local-context control supports a genuine whole-line coordinate.

## Audit order

Circularity → leakage → confounds → matched nulls → control fairness → measurement degeneracy → representation dependence → decision-rule fragility → audit completeness → interpretation.
