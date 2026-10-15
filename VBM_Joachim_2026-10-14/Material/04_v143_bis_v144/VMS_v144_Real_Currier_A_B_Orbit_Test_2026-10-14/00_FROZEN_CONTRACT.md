# Real-Voynich shared-operator admission contract (2026-10-14)

## RETRACTIONS/PREVIOUS NEGATIVES
- v144 orbit association is synthetic; exact autonomous equivalence prevents plaintext attribution.
- v143 visible suffix proxy did not discriminate carry from unchanged v142b.
- Prior CF3 planted-equivalence retrieval 0/4 in Herbal A and Q13; cannot claim a generic family-recovery instrument qualified by default.
- NK3 family-transfer, BR2 and OCC2 negative results remain. Whole-manuscript/section-mixed analyses prohibited.

## Prior identification gate
Candidate is a **surface** shared operator: an identical interior single-character substitution x↔y across at least three independently attested word shells, with common influence of a preceding word's terminal bigram, beyond per-shell and exact-token CONNECT predictors. A compact globally shared transformation must generalize **to new word shells**, not merely explain repeated known words. This is neither the full v143 cell grammar nor plaintext identification. The final two EVA characters are an *approximate observable* chosen in advance, not assumed to be the v144 bridge.

Competing mechanisms: H1 one shared suffix-conditioned substitution response across multiple shells; H0a independent shell-specific changes, H0b ordinary previous-word/whole-token CONNECT, H0c section+position conditioned lexical selection, H0d autonomous state machine implementing the exact same H1 probability law. H1 vs H0d is *mathematically nonidentifiable*, and a positive H1 score CANNOT identify plaintext source.

Extraction: IVTFF Zandbergen-Landini ZL3b (13 May 2025), strict P running-text lines, original folio metadata $L and $I, exclude uncertain raw tokens and do not connect neighbours across discarded tokens, exclude line crosses. Sections: Herbal A, Pharma A, Herbal B, Bio B, Stars B. No labels. Report exclusions and number of independent folios/quires. Use raw EVA first (representation alternative: raw reverse-atom EVA, predeclared). For within-section tests hold out entire quires when >=4; when only one quire exists hold out entire folios and flag bifolium dependence; do NOT pool A and B for primary effect.

Admissibility (before scoring): 1-edit interior shells of length 4..10 with exactly two observed variants differing by a single internal character, count >=8 total with at least three minority cases, and each shared ordered pair appears in >=3 shells; learn shells and pair operator frequencies on training only. Do NOT pick effect-favourable families. If section has <200 eligible events or <4 independent heldout units, mark unqualified. No training on holdout outcomes. Family extraction overlapping sites must count a given token once via fixed lexicographic tie-break; sensitivity must verify overlap.

Predictive score: held-out log2 likelihood of selected variant with hierarchical smoothing: per-shell base + shared (substitution pair, previous suffix bin) signal; compare against (i) per-shell base alone (ii) capacity-matched independent per-shell state model (iii) previous exact-token CONNECT; compare equal event positions. For held-out novel shells, base proportions cannot be learned: either specify *a priori* orientation baseline 0.5, or use training families with held-out folios and out-of-family operator test as a second split. A claim of shared transfer requires both shell-family separation and folio separation; absence of that makes result descriptive.

Controls in order: circularity, leakage, confounds (hand/quire/line position/token length), matched nulls (independent within-shell sign rotations and previous-state shuffles within folio/line-position buckets), equal-parameter fairness, degeneracy and planted positive control with prespecified mild injection, representation sensitivity, decision-rule fragility, audit completeness. Require planted control to beat hard null >=2 SD; original data advantage >=2 matched-null SD, each primary section separately, plus >=4 heldout units positive, and meaningful heldout effect over strongest nuisance. No model upgrade if control fails. Test both classes separately and do not infer plaintext.

If capacity or identifiability conditions cannot be met, stop and report admission failure rather than fitting additional families or changing thresholds. Preserve atomic pickle stage checkpoints and prominent retractions.
