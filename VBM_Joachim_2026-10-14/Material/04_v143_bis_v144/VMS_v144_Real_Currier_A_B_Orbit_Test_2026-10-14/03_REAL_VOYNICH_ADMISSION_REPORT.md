# Real-Voynich shared-transformation admission review
**14 October 2026 | final status: NO-GO for a shared A/B mechanism; Stars B raw-EVA signal is descriptive and representation-fragile**

## RETRACTED, NARROWED, AND FAILED FINDINGS — READ FIRST

1. **NOT PROMOTED:** The preliminary Stars B positive under *raw EVA* is not a qualified common operator. It fails the predeclared alternate representation: REVA leaf-held-out +0.003421 bits/eligible word above arbitrary-group null, SD 0.002145 (1.59 SD); only 4/12 positive leaf units. The raw-EVA result survives two local controls but cannot justify a representation-independent mechanism.
2. **NOT EXCLUSIONARY:** Negative Herbal B and Biological B experiments are underpowered for the predeclared 15% plant: 1.46 SD and 0.88 SD. Their observed negative scores are not evidence that cross-word transmission is absent.
3. **PHARMACEUTICAL A NOT TESTABLE at this resolution:** Although full-data descriptive filtering counted 193 eligible events, training-only family discovery under quire holdout reduces the test sample to just 20 events in one fold. Full-corpus selection is leakage; no relaxation permitted.
4. **OLDER RESULTS UNCHANGED:** Synthetic v142–v144 engineering claims, exact autonomous equivalent, the old B-only surface failure, NK3 family transfer negatives, CF3 planted retrieval failures and earlier cross-space tests are not overturned.
5. **MECHANISM NOT IDENTIFIABLE:** Even successful surface operator structure cannot distinguish plaintext-driven state from a fully autonomous surface transducer. The final two EVA characters are a *proposed, non-validated proxy*, not an independently observed v143 vowel bridge.

## Provenance and scope

Primary source: Zandbergen–Landini IVTFF 3b transcription, version 13 May 2025, file `eva_zl3b.txt` in [Aspect-Research/voynich-autoexploration](https://github.com/Aspect-Research/voynich-autoexploration/blob/master/data/transcriptions/eva_zl3b.txt), Git blob SHA `2a4533ab9bdfa85db9bad602d590978953055df1` (returned by the GitHub connector). Data were read directly through GitHub, not copied from the synthetic v143 reference. The original full source text **is not part of this local package**. It can be independently retrieved using that revision and URL. Source byte length reported by the connector 411,671 characters. A second independent transcription was **not** run here.

Only lines carrying paragraph-running P loci and folio metadata `$L` A/B and `$I` H/P/B/S were included. Uncertain readings (`[...]`, `{...}`, `<->`, glyph uncertainty) were replaced by separators; only uninterrupted lowercase `[a-z]+` tokens included. Consecutive-word pairs were **within one source line and contiguous**, never joined across excluded tokens, figures, line or folio boundaries. The five sections were Herbal A, Herbal B, Pharmaceutical A, Biological B and Stars B. Labels, radial text, and other section identifiers were excluded.

## Mechanism and identifiability gate

One shared edit-operation label is the ordered substituted interior character pair (e.g. `ao`, `eo`, `kt`). Word families are all words of length 4–10 identical except for one *internal* character. Only exactly two observed variants per family, at least eight training occurrences, at least three occurrences of the minority variant and at least three independently attested shells per edit-operation label qualify. Family definitions are discovered on training material ONLY for each physical holdout. At each eligible held-out position, the previous word's last two EVA characters give a small train-frozen state inventory (six frequent states plus OTHER).

The shared operator estimates a state-dependent correction to each family's baseline variant odds from other families with the **same substitution label**, explicitly leaving out the current family when estimating its shared correction. Countermodels include the family-only prior, independent per-family state corrections, and exact preceding-word corrections (same test events, smoothed), plus family-sign rotation and arbitrary edit-label regrouping. This is a narrow, key-independent *surface* test: it does not implement the full v143/v144 bridge grammar.

A competing autonomous surface process can exactly implement the same rule. Thus even a successful result would identify a surface regularity, not language or cipher. The only admissible outcome for manuscript mechanism attribution would require additional independent restrictions or plaintext anchors.

## Primary real-text held-out results

Training-only family maps rebuilt for each fold. Herbal A/B and Pharmaceutical A hold out entire **quires**. Biological and Stars are each a single quire, so hold out **folios** in the primary test (leaf-pair holdout for Stars in an additional bound). 120 random *whole-family sign rotations* preserve shell frequencies and previous-word contexts while destroying common orientation of substitution labels. Gain and effect are bits per eligible held-out event, null SD is sample SD of 120 rotations.

| Section | Eligible test events | Shared gain vs family-only | Effect relative to sign-rotation null | Null SD | Effect/SD | Positive units | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| Herbal A | 1,006 | −0.001582 | +0.003564 | 0.003637 | +0.98 | 3 / 9 | NO-GO |
| Herbal B | 307 | −0.033133 | −0.016660 | 0.012877 | −1.29 | 0 / 6 | NO-GO / power concern |
| Biological B | 1,226 | −0.009843 | −0.003814 | 0.003878 | −0.98 | 6 / 19 | NO-GO / power concern |
| Stars B | 2,057 | +0.007599 | +0.014614 | 0.003291 | +4.44 | 16 / 23 | *PRELIMINARY* positive only |
| Pharmaceutical A | 20 | n/a | n/a | n/a | n/a | 1 evaluable fold | UNQUALIFIED |

All effective contrasts are specific to the conditional word-family assay, not all manuscript tokens. They are not statistical evidence for a *cipher*.

## Bounding the preliminary Stars B positive

1. **Specific shared substitution labels vs arbitrary family groupings:** Permute which existing shells are assigned to substitution groups while keeping group-size inventory and model complexity. With 72 random assignments, raw EVA folio holdout gives +0.012420 bits/eligible word above that grouping-null, null SD 0.003094 (4.01 SD). Original shared gain +0.007599 bits. This grouping null preserves group counts, but *does not preserve all word-shape covariates*, so it is not a final hard matched historical control.
2. **Leaf-pair holdout:** Excluding both recto and verso for each folio number together, raw EVA gives +0.015171 bits/eligible word vs grouping-null SD 0.003803 (3.99 SD); 9/12 leaf groups positive. Validates that the raw score was not solely a folio-side split artefact. All Stars pages remain in one quire; no independent quire test is possible.
3. **Predeclared representation check:** In equivalent leaf-pair holdout using REVA compound-atom reversal, shared gain −0.001855 bits/eligible word; effect above arbitrary groups +0.003421, null SD 0.002145 (1.59 SD), 4/12 leaf groups positive. Compared with +0.014120, SD 0.003588 (3.93 SD), 9/12 positive under raw EVA in that same separate sensitivity run. *Different eligible word populations* (REVA 3,016 vs EVA 2,065), so no paired causal difference can be claimed. This **fails** the mechanism promotion criterion for representation robustness.
4. **Positive qualification at 15% injection:** A known shared operator was implanted by replacing the binary within-family choice for 15% of eligible positions with a choice tied to the previous token suffix ending `y`, without adding new word forms. This tests sensitivity, **not manuscript truth**. 60 family-sign rotations. Herbal A +0.020818 bits/event, null SD 0.006994 (2.98 SD), 6/9 positive; Herbal B +0.017484, SD 0.012012 (1.46 SD), 3/6 positive; Biological B +0.003764, SD 0.004261 (0.88 SD), 9/19 positive; Stars B +0.022996, SD 0.003812 (6.03 SD), 15/23 positive. Thus only Herbal A and Stars B have adequate detection of this particular planted perturbation. The other two negatives are not exclusionary.

## Required audit order

1. **Circularity:** Full-data candidate counts were explicitly for feasibility only and never used as trained predictor families. Actual predictions use leave-one-quire/folio-out family discovery, suffix categorisation and fitted operators. The exploratory Stars finding was then bounded, not secretly refitted.
2. **Leakage:** All candidate-shell observations on held-out quire/folio excluded from learning that fold's families. Predecessor contexts taken only from the same intact line; uncertainty removals break adjacency. Within-quire Stars/Bio still leave dependence across folios.
3. **Confounds:** section is never pooled. Currier A/B, hand, section, quire, page, word morphology, word length, positional differences remain confounded. Stars one quire; no general manuscript inference.
4. **Nulls:** 120 matched family-sign orientations for initial cross-family predictor; 72 edit-label reassignments for Stars specificity bounds; 60 for implanted qualification. First null preserves shell counts and contexts, second preserves number of edit-group labels but not full geometry. No null SD from an independent manuscript sampling distribution.
5. **Control fairness:** Family base, independent state and exact preceding token compared at identical eligible positions. Arbitrary-label grouping uses same family data and parameter count; morphology matching incomplete and declared.
6. **Degeneracy:** 15% planted operator rejected in Herbal B and Biological B due weak sensitivity; those negatives cannot justify an exclusion. Pharma A lacks a testable dataset at frozen threshold.
7. **Representation:** Raw EVA Stars positive collapses below 2 SD under REVA atom-reversal, with different admitted samples. No representation-invariant positive established.
8. **Decision rule:** Precommitted promotion requires ≥2 null SD, ≥4 positive holdout units, stronger than nuisance, and both Currier classes / representation stability. Fails explicitly. No posthoc threshold adjustment.
9. **Audit completeness:** Preserved contract, scope counts, bounds contract, JSON results and atomic pickle checkpoints. Main numerical scoring was run in the connected GitHub computation environment, not in this offline container; package does not assert an independent local replay of the entire 411,671-character primary input.

## Decision and research value

**No promotion of v143/v144 as an empirically identified Voynich-wide mechanism.** A narrowly defined, pooled cross-family word-variant effect was found in *raw-EVA Stars B* and survived a grouping null and leaf holdout, but it fails alternate representation and does not generalise to Herbal A. The realistic takeaway is an unresolved section-specific surface signal and a quantified limit on the instrument; it is neither evidence for original VBM nor evidence against plaintext encoding.

Do **not** iterate another generator or widen family criteria to rescue the failed results. A future test would require an independently fixed glyph/FORM segmentation and a discriminating, representation-stable prediction qualified on both A and B under hard section/hand/context nulls. Until then this branch remains **NO-GO**. Other established FORM/CONNECT/LINE_ENTRY results are unaffected.
