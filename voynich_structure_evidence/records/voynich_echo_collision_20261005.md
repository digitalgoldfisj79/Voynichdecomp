# Voynich Echo rule collision test — 2026-10-05

- key: `voynich_echo_collision_20261005`
- status: completed
- created: 2026-10-05T18:37:37.60134+00:00
- updated: 2026-10-10T16:27:25.988447+00:00

---

# RETRACTED / SUPERSEDED FINDINGS — READ FIRST
- RETRACTED AS FAMILY-SPECIFIC EVIDENCE: the weak-control upper bound suggested Echo-family conditioning closed ~80–93% of the Stolfi gap with ZLZI/ZLZB/TTLI family-vs-random-graph effects at ~2.92–3.36 SD. That control fixed first FORM piece and exact raw length but did NOT fully match morphology. Do not cite it as a promoted result.
- NOT ROBUST: the hierarchical hostile-control result remained >2 SD in ZLZI and TTLI but ZLZB was 1.91 SD. Under the strictest L0 control (piece count + final K8 + ED bin + gallows occupancy, in addition to first piece + exact raw length), all three transcriptions fall below 2 SD.
- Therefore no Echo family-specific SELECT→FORM layer is promoted.

# ECHO RULE COLLISION TEST — 2026-10-05

Status: COMPLETE / CANDIDATE INTERESTING, PRIMARY PROMOTION FAILS.

## External candidate
Public repo: eo777/voynich-rules, version 2026.10.05.
Conservative frozen ruleset: published/supported core represented in fit_data only:
ed, d, B1, C1, C3, C4, S1, S4.
Excluded from primary: S2/S3 experimental/default distinctions and all catalogue-only speculative rules.

Core graph:
- 2,362 directed edges
- 3,401 node forms
- 1,162 connected components
- median component size 2
- largest component 44
Coverage of paragraph-token occurrences:
- ZLZI 11,604 / 33,698 = 34.44%
- ZLZB 11,610 / 33,703 = 34.45%
- TTLI 11,083 / 30,843 = 35.93%

## Firewall / physical split
Canonical physical-bifolium folds reproduced from hf_emergent_occupancy_fold.py.
Discovery/control metadata: folds 2/3.
Validation/general train where needed: fold 4.
Untouched evaluation: folds 0/1.
ZLZI canonical event count reproduced: 34,087.

Important limitation: Echo rule discovery itself was performed on the Voynich corpus and the public repository is generated from an unpublished workspace. Our folds are untouched by OUR fitting, but not guaranteed untouched by Echo's original rule discovery. This is an external-candidate diagnostic, not a prospectively independent replication.

## Test 1 — Directionality: Echo source→target should shift A-like forms toward B
Population: Echo earlyA vs HB regime metadata.
Effect: heldout log2-odds(HB/earlyA) target minus source, aggregated by connected family.
Matched legal-edit null preserves target length delta, ED bin, first/last-change geometry, plus discovery frequency and A/B-propensity.

Strict primary (freq/prop bins ±1):
- ZLZI combined: 5 families; observed 3.835 bits; matched-null SD .553 gives large edit-null contrast, BUT family-orientation sign-null SD 2.160 => z_sym 1.776. THE METRIC DOES NOT RESOLVE DIRECTIONALITY.
- TTLI combined: observed 3.427; orientation-null SD 1.795 => z_sym 1.909. THE METRIC DOES NOT RESOLVE DIRECTIONALITY.

Relaxed sensitivity (freq/prop bins ±2):
- ZLZI: 9 families; z_sym 2.526.
- TTLI: 8 families; z_sym 2.410.
But individual untouched folds remain <2:
ZLZI fold0 1.368, fold1 1.635; TTLI fold0 1.378, fold1 1.416.

Adjudication: directional A→B interpretation NOT promoted. Pooled relaxed evidence is suggestive only.

## Test 2 — External-context equivalence
Context vectors use heldout previous/next first glyph, final glyph, neighbor length, and line position.
Primary null fixes edit geometry, discovery frequency ±1 and same dominant regime.

ZLZI primary combined:
effect +.00613 cosine; null SD .02500 => 0.245 SD. THE METRIC DOES NOT RESOLVE THIS.
TTLI primary combined:
effect +.08915; null SD .04311 => 2.068 SD, but only 4 families.

Relaxing regime match:
- ZLZI combined +.04820 / .01348 = 3.576 SD
- TTLI +.07428 / .01543 = 4.813 SD
But untouched folds individually fail:
ZLZI 1.758 / 1.662; TTLI .863 / 1.182.

Adjudication: external-context equivalence NOT promoted. Signal is sensitive to regime-matching and pooling.

## Test 3 — Whole-token Stolfi/Mauro closure upper bound
Frozen FORM reconstructed from current programme:
~32-piece segmentation, K8 continuation grouping, exact-piece+depth STOP, 99% continuation crib, gallows-used rho=.25.
FORM fit folds2/3/4; final folds0/1.
Population: completely unseen test tokens.
All generated comparisons fix observed first FORM piece and exact raw character length.
Echo oracle candidate: another unseen graph node in same Echo connected component.
This is an oracle upper bound, not a deployable selector.

Weak-control coverage:
ZLZI paired 125/2106 novel = 5.94%.
ZLZB 126/2107 = 5.98%.
TTLI 99/1771 = 5.59%.
Weak random-graph controls suggested strong Stolfi closure but are retracted as under-matched.

Hierarchical hostile match additionally controls piece count and ED in all cases, adding final K8 and gallows occupancy wherever support allows.
Coverage fell to:
- ZLZI 56 pairs (2.66% all novel)
- ZLZB 52 (2.47%)
- TTLI 46 (2.60%)

Stolfi family-specific effect vs hierarchical matched control:
- ZLZI +.125 / null SD .05820 = 2.148 SD
- ZLZB +.09615 / .05034 = 1.910 SD — THE METRIC DOES NOT RESOLVE THIS.
- TTLI +.15217 / .05719 = 2.661 SD
Cross-transcription gate fails.

Mauro on the selected population is saturated (real/family ~.99–1.00) and family-specific effects are ~1 SD or non-estimable. It provides no independent discrimination.

## Bounding test — strictest L0 morphology-matched control
Control matches:
- first FORM piece
- exact raw length
- exact piece count
- final K8 class
- edit-distance bin to the real target
- gallows occupancy bin
and excludes the target Echo component.

Strict-L0 Stolfi family-specific result:
- ZLZI n=27: effect +.148148; null SD .091106; z=1.626.
  fold0 1.338 SD; fold1 1.000 SD.
- ZLZB n=26: effect +.076923; null SD .054292; z=1.417.
  fold0 ~1.000; fold1 ~1.000.
- TTLI n=21: effect +.047619; null SD .047631; z=1.000.
  fold0 ~1.000; fold1 non-discriminatory.

THE METRIC DOES NOT RESOLVE A FAMILY-SPECIFIC STOLFI ADVANTAGE UNDER THE STRICTEST FAIR CONTROL.

## Audit
Circularity: material. Echo rules are derived from Voynich A/B distributions.
Leakage: public discovery workspace unpublished; physical folds are only clean relative to our test, not necessarily Echo discovery.
Confounds: morphology/edit geometry is major; strict matching removes the apparent family-specific closure.
Matched nulls: performed with declared relaxation ladder; strictest L0 controls all fail <2 SD.
Control fairness: weak graph control retracted; strict L0 is controlling adjudication.
Measurement degeneracy: Mauro saturates near 1.0 on selected forms, so it cannot discriminate family specificity.
Representation dependence: relaxed results broadly reproduce across ZLZI/ZLZB/TTLI; strict L0 fails in all three.
Decision-rule fragility: apparent positive result disappears as control matching tightens.
Audit completeness: cannot audit unpublished Echo discovery workspace; no causal promotion allowed.

## Final adjudication
SUPPORTED:
1. Echo has identified a substantial, nontrivial network of real Voynich morphological correspondences.
2. These correspondences are strongly aligned with Currier A/B redistribution under relaxed matched-edit tests.
3. Echo families preferentially contain Stolfi-conformant forms, but this is largely explainable by the same morphology/edit geometry used to define the neighborhoods.

NOT SUPPORTED:
1. robust A→B transformation direction;
2. context-equivalent latent complete-form families after strict controls;
3. an independent Echo SELECT→FORM family layer;
4. family-specific closure of the frozen-FORM Stolfi/Mauro residual.

Interpretation:
Echo is useful as an externally specified MORPHOLOGICAL CANDIDATE GENERATOR / correspondence graph. It is not presently evidence for a new causal production layer. Its strongest apparent collision with the SELECT→FORM residual collapses under strict morphology-matched controls.

Recommended use:
retain Echo edges as an external feature set for future exact/finer-grained SELECT tests, but do not add them to the Super-Grammar as causal rules and do not reopen a broad Naibbe-style family layer on this evidence alone.

