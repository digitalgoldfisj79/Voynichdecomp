# Voynich context-first CF0 calibration closeout — 2026-10-05

- key: `voynich_context_first_CF0_20261005`
- status: active
- created: 2026-10-05T20:18:11.830714+00:00
- updated: 2026-10-10T16:27:25.988447+00:00

---

# CONTEXT-FIRST LEXICAL EQUIVALENCE — CF0 CALIBRATION CLOSEOUT
Date: 2026-10-05
Status: CALIBRATED-IMPERFECT / PROCEED WITH FROZEN RULE
Preregister: voynich_context_first_lexical_equivalence_preregister_20261005

Scientific correction:
The original >=80% same-source precision gate was an arbitrary governance threshold, not a scientific discontinuity. The final frozen calibration reaches 11/14 = 78.57% true same-source pairs at the prespecified q=.995 shared-fingerprint cutoff, while the shuffled-order control gives 1/7 = 14.29%. The difference between 78.6% and 80% is not scientifically meaningful at n=14. Therefore the assay is not treated as "certified truth", but it is sufficiently informative to proceed conditionally, with the rule frozen and false-positive behaviour explicitly measured in Voynich nulls.

Valid completed runs:
1. HF 6ac4039cfbc85ba6823ac66b — initial pooling assay.
   Ordered: +0.03363 bits/event, 3.965 block SD; random-partition z 3.179; both heldout halves positive; exact-token control 5.660 SD.
   But only 2/6 selected pairs were true same-source. Shows real context signal but semantic/syntactic-neighbour confound.
2. HF 6ac40434fbc85ba6823ac6c3 — diagnostic.
   Same-source discrimination: context cosine AUC .833; pooling gain AUC .706; near-context AUC .755.
3. HF 6ac405f8404719ba3765895d — split-half fingerprint.
   Ordered cross AUC .734 / gap .741; shuffled .463/.479. Top 1% candidate set 56.5% true.
4. HF 6ac40f07fbc85ba6823acdf5 — final shared-context + conditional-indistinguishability calibration.
   q=.995 shuffled-fingerprint threshold:
   27 ordered candidates -> 14 selected by heldout discrim_gain<=0 -> 11/14 = 78.57% true same-source.
   Stricter discrim_gain<=-.005: 13 selected -> 76.92% true.
   Shuffled source order: 14 candidates -> 7 selected -> 14.29% happen to be same-source.
   This is the frozen real-data rule: q=.995 contextual-fingerprint threshold relative to matched/shuffled null + fold4 conditional indistinguishability <=0.

Invalid/error runs, no science:
- 6ac405befbc85ba6823ac79b diagnostic indexing error.
- 6ac4067e404719ba376589ad empty-summary error.
- 6ac40de4fbc85ba6823acd57 404 from blob SHA.
- 6ac40e0efbc85ba6823acd69 old code repeated empty-summary failure.
- 6ac40ede404719ba37658ee9 pair-debug only; identified int64 sparse-index incompatibility.

Revised inference discipline:
- Do NOT tune q beyond .995 on the same synthetic truth.
- Do NOT call every selected pair a true equivalence; synthetic precision is ~79%, not 100%.
- In real Voynich, compare aggregate selected-pair behaviour to matched label-shuffle and matched random-pair nulls.
- Final folds0/1 remain untouched for generalisation.
- Morphology/FORM/ED/DINO remain firewalled until pairs are frozen.
- A real positive is an aggregate excess of contextually exchangeable pairs plus heldout generalisation, not any anecdotal pair.

Next phase:
Run CF1/CF2 on folds2/3->4 using frozen q=.995 criterion and real matched null; then CF3 once on folds0/1. Only if aggregate CF3 is positive inspect pair surface structure and consider CF4.

