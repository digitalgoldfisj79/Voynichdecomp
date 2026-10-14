# Voynich context-first lexical equivalence test — preregistration — 2026-10-05

- key: `voynich_context_first_lexical_equivalence_preregister_20261005`
- status: active
- created: 2026-10-05T20:02:03.754625+00:00
- updated: 2026-10-10T16:27:25.988447+00:00

---

# VOYNICH CONTEXT-FIRST LEXICAL EQUIVALENCE TEST
Date: 2026-10-05
Status: PREREGISTERED / NOT YET RUN
Programme lineage: post-NK3 correction; not NK4.
Authoritative key: voynich_context_first_lexical_equivalence_preregister_20261005

# 0. MOTIVATION

Corrected NK3 rejects every tested SURFACE-FIRST family construction:
- handcrafted K12 FORM family;
- learned FORM embedding;
- FORM+ED hybrid;
- pure ED;
- raw DINO visual families;
- FORM-residualized DINO families.

Most effects are strongly negative on both untouched folds, often 6-17 physical-block SD below zero.

However the exact-token positive control is real in ZLZI/ZLZB:
EXACT_TOKEN_TOP24 context gain ~+.0046-.0047 bits/token, block z ~11.4, both final folds positive, context-shuffle z ~5.
Thus external context contains some exact lexical/token identity information even though broad surface-family identity does not.

The next test reverses the causal search direction.

Rejected NK3 direction:
SURFACE SIMILARITY -> proposed family -> ask whether family is context-equivalent.

New direction:
EXTERNAL CONTEXT -> learn lexical equivalence -> freeze classes -> only then inspect surface realisations.

No current-token spelling/morphology/ED/DINO feature may enter class discovery, tuning, or primary inference.

# 1. SCIENTIFIC QUESTION

Do distinct exact Voynich token types occupy statistically exchangeable external-context niches, such that pooling their context distributions improves or preserves prediction on untouched physical material beyond frequency/section/page ecology?

If yes:
Do the context-derived equivalence classes subsequently predict surface/FORM properties of completely novel token types when family is inferred from context alone?

This tests a weak Naibbe-like grain without assuming homophones are morphologically similar.

# 2. HYPOTHESES

H0 EXACT-LEXICAL:
External context distinguishes exact token types; any nontrivial merge loses heldout information once nuisance and shrinkage are controlled.
Interpretation: SELECT is near-exact lexical/fine-form choice.

H1 CONTEXT-EQUIVALENT CLASSES:
Two or more exact token types share an externally exchangeable niche. A class-level model generalises on heldout physical material and yields legitimate token merges not explained by section/frequency/page ecology.

H2 SOURCE-TO-SURFACE BRIDGE:
Context-derived class inferred WITHOUT observing the current token predicts the current token's whole-form properties, including on exact types absent from discovery.
This would support:
source/context state -> lexical equivalence class -> productive surface realisation.

# 3. FIREWALL

Current token information forbidden during family discovery:
- EVA spelling/string;
- BPE pieces;
- K12/K8 FORM states of CURRENT token;
- current token length;
- gallows;
- ED;
- DINO/image embedding;
- Stolfi/Mauro/Zattera labels;
- section-specific morphology;
- exact-token frequency may be used only for eligibility, weighting and matched nulls, not as a clustering feature.

Exact token ID is allowed only as the row/key whose EXTERNAL context distribution is being estimated.

External-context features may use OTHER tokens:
Primary local window:
- exact neighbour token identity at lags -2,-1,+1,+2, restricted within physical line; never bridge line boundaries;
- neighbour coarse START K12 / FINAL K12 / piece length as secondary channels;
- previous/next physical-line opener is excluded from primary to keep LINE_ENTRY separate.

Nuisance / stratification:
- current section;
- Currier/register where independently available;
- line-position bin;
- paragraph-depth bin;
- page/folio identity only through matched/permutation controls, not a predictive family feature;
- discovery-only token frequency and section propensity for weighting/null matching.

Primary population:
strict +P0 ordinary running-text tokens;
physical line-start current tokens excluded from primary because LINE_ENTRY is separate.
ZLZI primary; ZLZB and TTLI replication.

Physical folds:
discovery = 2,3;
validation = 4;
final untouched = 0,1.
No retuning after fold4.

# 4. CF0 — ASSAY CALIBRATION BEFORE REAL VOYNICH

Purpose:
Prove the context-first assay can recover true same-source equivalence when same-source surface variants are deliberately NOT morphologically compact.

Synthetic source:
- Circa Instans ordinary prose;
- notation-like structured sequence from NK2;
- shuffled-order hostile control.

Renderer:
use stripped NK1/NK2 source->surface channel but RANDOMISE/ORTHOGONALISE surface family geometry so variants of the same source identity are deliberately spread across FORM/ED space.
Preserve source frequencies, token lengths and frozen local FORM.

Run the exact context-first learning algorithm below without access to planted source labels.

Calibration success:
- context-derived pair/family relation recovers planted same-source equivalence above shuffled-order control;
- positive source posterior/next-source information survives;
- surface morphology similarity is at matched-random level by construction;
- hostile shuffled source order fails.

If CF0 cannot recover planted non-morphological equivalence at a coupling comparable to ordinary-prose NK2, DO NOT interpret a real-Voynich null as evidence against equivalence. Improve assay/power first.

# 5. CF1 — CONTEXT SIGNATURE LEARNING ON DISCOVERY ONLY

Eligibility:
Primary threshold initially fixed at >=20 discovery occurrences across folds2/3 and >=3 validation occurrences.
Final inference additionally requires >=3 occurrences across folds0/1.
Frequency sensitivity thresholds 10, 30, 50 are secondary only; no threshold may replace primary after outcome inspection.

Context vocabulary:
Top 256 exact neighbour token types by discovery frequency plus OTHER, separately by lag -2,-1,+1,+2.
No heldout frequency information selects the context vocabulary.

Occurrence-level nuisance residualization:
For each lag/context outcome c, fit discovery-only baseline P(c | section, line-position, paragraph-depth, optional Currier/register with hierarchical smoothing).
For each occurrence form residual/log-likelihood contribution relative to that nuisance baseline.
Aggregate residual context distributions by CURRENT exact token type.

Primary representation:
concatenate four lag-specific nuisance-residual context profiles.
Apply discovery-only variance stabilization / shrinkage.
Dimension reduction: randomized SVD on discovery matrix.
Candidate embedding dimensions D in {8,16,32,64}.
D is selected on fold4 only by heldout context likelihood, never by morphology.

Secondary context representation:
paragraph-level co-occurrence residual profile, separately tested; may not replace primary local-window result.

# 6. CF2 — LEARN MINIMAL EQUIVALENCE RELATION

Do NOT KMeans current morphology.

Primary algorithm:
1. Each eligible exact token starts as its own lexical node.
2. On discovery embeddings, propose only the top 20 nearest candidate partner types per token by cosine similarity.
3. Candidate pair merges are evaluated on fold4 by predictive exchangeability:
   compare heldout external-context codelength under
   A) separate token-specific context distributions;
   B) one pooled shared context distribution,
   with identical hierarchical Dirichlet smoothing fitted on discovery.
4. Pooling benefit is defined as L_separate - L_pooled in bits/context-event.
5. A proposed merge is eligible only if:
   - validation pooling gain > 0;
   - both left-context and right-context directional panels are non-negative;
   - gain exceeds the 99th percentile of frequency + section-propensity matched random pair merges on fold4.
6. Resolve overlapping accepted pairs by validation-likelihood agglomeration:
   iteratively take the merge with largest positive validation gain;
   recompute candidate family pool;
   stop when no remaining merge passes the frozen criterion.
This yields a context-derived equivalence graph/family partition with no current-token surface information.

Complexity control:
Matched random mergers use identical family-size distribution, discovery frequencies and section propensities.
A family is not licensed merely because pooling low-frequency types reduces variance.

# 7. CF3 — FINAL UNTOUCHED CONTEXT TEST

Freeze:
- eligibility;
- context vocabulary;
- nuisance model family;
- SVD D;
- proposed merges;
- final family partition;
- smoothing.

Evaluate folds0/1 once.

Models:
B0 nuisance only.
B1 exact-token context model with hierarchical shrinkage.
B2 context-family only.
B3 hierarchical family + exact-token deviation.

Primary statistics:

A. FAMILY INFORMATION ABOVE MATCHED RANDOM
B2 codelength gain over B0 minus same statistic under matched random family partitions.

B. HIERARCHICAL POOLING
B3 heldout codelength gain over B1.
This asks whether family-level sharing adds generalisable information beyond independent exact-token estimates.

C. RECOVERY FRACTION
(B2-B0 gain) / (B1-B0 gain), where denominator positive.
Calibration CF0 defines expected range for true equivalence; do not invent a real-data threshold after inspection.

D. PHYSICAL REPLICATION
Report fold0 and fold1 separately; physical bifolium block mean/SD together with effect size.

Primary promotion requires:
1. A >2 matched-null/block SD;
2. positive direction in both final folds;
3. matched random family controls fail;
4. SECTION_ONLY and page-frequency family controls fail;
5. B3 is non-negative versus B1; >2 SD is strong evidence but not mandatory if family-only A is independently decisive and CF0 calibration shows finite-sample saturation of B1;
6. same-direction ZLZI/ZLZB; TTLI reported separately if exact-token positive control remains representation-specific.

Exact-token positive control:
B1 must reproduce a positive exact-token contextual signal in ZLZI/ZLZB. If it does not, assay recertification fails and no family conclusion is allowed.

# 8. POST-FREEZE SURFACE INSPECTION — NOT PART OF FAMILY LEARNING

Only after CF3 family labels are irrevocably frozen:

Measure within-family surface similarity:
- frozen FORM path/K12/K8;
- raw EVA edit distance ED1/2/3;
- DINO visual distance;
- length;
- gallows occupancy;
- Stolfi/Mauro diagnostic coordinates.

Compare each to matched random token sets preserving:
frequency, section propensity, raw length and family size.

Possible outcomes:
A. Context families are also morphologically compact:
surface and context structure coincide; reconsider why NK3 surface-first assay failed.

B. Context families are NOT morphologically compact:
this is the genuine non-obvious homophony/codebook-like case NK3 could never find.

C. No nontrivial context families:
supports fine-grained/exact lexical SELECT rather than broad homophony.

Surface compactness alone is never a promotion criterion.

# 9. CF4 — MASKED NOVEL-TYPE BRIDGE [DECISIVE SECOND STAGE]

Question:
Can a family inferred from context alone predict the form of a token whose exact type was never seen in discovery?

Population:
Final folds0/1 occurrences whose exact EVA token type has ZERO occurrences in folds2/3/4.
Current token is fully masked for inference.
No token identity, spelling or current morphology is used to infer family.

Family posterior:
P(family | external context, nuisance)
using frozen CF3 context models.

After family posterior is computed, reveal current token only for scoring.

Discovery-trained family surface distributions:
For each frozen context family, estimate distributions over:
- current START K12;
- FINAL K12;
- piece-count bucket;
- gallows occupancy;
- complete frozen FORM path summary;
- Stolfi diagnostic signature;
- Mauro diagnostic signature.
These distributions are estimated only from discovery/validation member types AFTER family labels were frozen by context-only learning.

Primary novel-type score:
posterior-weighted family surface likelihood versus nuisance-only surface likelihood conditioned on section and exact raw length where appropriate.

Critical complete-token bridge:
test whether context-inferred family reduces the previously observed Stolfi/Mauro novel-token residual relative to the frozen FORM baseline WITHOUT using Stolfi/Mauro in family inference.

Promotion requires:
- >2 physical-block SD on untouched novel-type occurrences;
- positive both final folds where powered;
- matched context-shuffle and family-label permutation nulls;
- no gain when family posterior is replaced by section/page-only posterior.

This is the strongest source->family->surface test in the programme because the token form is unseen and masked at family inference time.

# 10. NULLS / HOSTILE CONTROLS

Required:
1. Within-page + section + line-position current-token-label shuffle preserving exact token counts.
2. Frequency + section-propensity matched random token merges.
3. Section-only families.
4. Page-prevalence-only families.
5. Random families with identical family-size distribution.
6. Morphology-derived NK3 families as negative historical comparators, not retrained positives.
7. Exact-token identity positive control.
8. Shuffled source-order synthetic CF0 negative control.

Max-stat / multiplicity:
Validation handles candidate merge selection.
Final inference uses aggregate frozen-family statistics and family-label permutation nulls; do not cite the best individual pair as a manuscript-wide result without max-stat correction.

# 11. OUTCOME INTERPRETATION

Outcome 1:
Exact-token positive, context families fail.
=> strongest support for fine-grained/exact lexical SELECT. Naibbe-style broad equivalence further demoted.

Outcome 2:
Context families pass, but novel CF4 fails.
=> recurrent token types share contextual niches, but no evidence that the niches specify productive surface form. Could be semantic/topic classes rather than homophony.

Outcome 3:
Context families pass and CF4 predicts novel surface/whole-form structure.
=> major positive: an upstream context/source equivalence class predicts surface realization without current-token morphology. This is the cleanest evidence yet for a source->family->FORM bridge.

Outcome 4:
Context families are surface-dissimilar yet CF4 passes.
=> strongest Naibbe-grain result: hidden equivalence exists but is not a surface morphology family.

Outcome 5:
Only section/page ecological classes pass.
=> no lexical equivalence; effect belongs to repertoire ecology.

Outcome 6:
Exact-token positive control itself fails under this stricter framework.
=> revisit prior exact-token context result before drawing any lexical conclusion.

# 12. WHAT THIS TEST DOES NOT DO

It does not:
- decode tokens;
- assume source language;
- assume plaintext;
- assume notation;
- assume homophony;
- use Naibbe codebook;
- use cards/respacing;
- restore PGCS/P70;
- alter frozen FORM;
- infer semantics from context clusters.

# 13. EXECUTION PLAN

Stage A: CF0 synthetic non-morphological-homophone calibration.
Stage B: CF1 discovery embeddings.
Stage C: CF2 validation merge selection.
Stage D: CF3 untouched family test.
STOP and adjudicate.
Only if CF3 has nontrivial context classes: CF4 masked novel-type bridge.

Heavy pairwise merge/permutation work should run on Hugging Face.

Per-phase Supabase keys:
voynich_context_first_CF0_20261005
voynich_context_first_CF1_CF2_20261005
voynich_context_first_CF3_20261005
voynich_context_first_CF4_20261005

Do not write NK4 confirmatory records unless this independent context-first programme first passes CF3.


## 2026-10-05 EXECUTION AMENDMENT — CF0 ASSAY REPAIR BEFORE REAL DATA
The preregistered pooling-gain merge criterion failed CF0 synthetic truth and was retracted before any real Voynich evaluation. It selected contextually similar different source words because pooling rewards shrinkage.
Replacement was developed entirely on synthetic truth:
independent-discovery-fold context exchangeability graph, q99.5 matched-random threshold, retaining overlapping edges; final graph-level null preserves degree/component topology through matched node relabelling.
CF0 multiseed ordered prose yields 19/19 true same-source discovery edges across five seeds; shuffled source order does not transfer. This repaired assay was frozen before real Voynich CF1-CF3.
Real result: voynich_context_first_CF3_20261005.


## CF0 POWER BOUNDARY — 2026-10-05
Full record: voynich_context_first_CF0_20261005.
Context-first exact-equivalence recovery FAILS at ordinary Circa-Instans prose strength: best frozen heldout panel rule gets 15/27=.5556 precision on untouched seeds despite shuffled-order control 0/31.
The identical frozen rule has overwhelming power for NK2-strength notation-like order: 120/120 true selected pairs across eight ordered seeds; shuffled source order 6/120=.05.
Therefore real-Voynich nulls cannot reject ordinary-prose homophony. A bounded real diagnostic can test for notation-strength equivalence without retuning.


## CF3N BOUNDED REAL RESULT — 2026-10-05
Full record: voynich_context_first_CF3N_20261005.
Using the notation-strength rule frozen by CF0, current-token morphology fully hidden during selection, the real Voynich test passes in all three transcriptions.
ZLZI pooled-vs-exact +.0102667 bits/event / block SE .0038874 =2.641 SD; both final folds positive; matched-random advantage2.610 SD.
ZLZB +.0103043/.0038936=2.646 SD; same ten frozen pairs as ZLZI; matched-random2.662 SD.
TTLI +.0092460/.0025899=3.570 SD; both folds positive; matched-random3.249 SD.
This is a context-equivalence positive only. It does not establish homophony/source identity. CF4 masked novel-type surface prediction is now licensed and must use frozen CF3N relations.


## 2026-10-05 CF0 CALIBRATION REVISION
The original >=80% precision gate is treated as an arbitrary governance threshold, not a scientific discontinuity. Final q=.995 calibration yields 11/14=78.57% true same-source versus 1/7=14.29% under source-order shuffle. Proceed conditionally with q=.995 frozen, explicit real-data nulls, and no further synthetic threshold tuning. Full closeout: voynich_context_first_CF0_20261005.

