# Super-Grammar v03 Cross-Domain Control Protocol XD1

**Freeze date:** 2026-09-20  
**Status:** FROZEN BEFORE EXTERNAL OUTCOMES  
**Parent release:** `supergrammar_v03_release_20260920`

## Purpose

XD1 is a prospective external-control projection of the certified v03 running-text grammar.

It is deliberately **not** the v03 theorem suite copied onto other corpora. Several v03 models condition on Voynich-specific section, Davis hand, Currier or particular glyph sequences. XD1 removes those manuscript-specific coordinates and reruns both Voynich and every external corpus through the same generic measurements.

The unit of comparison is a **profile**, not a scalar distance or winner. No omnibus similarity score, nearest-neighbour ranking or mechanism verdict is licensed by XD1.

## Admission rule

A corpus must be listed in `public.vms_supergrammar_control_registry_v03`. Only controls whose audit status permits the specific metric family in `public.vms_supergrammar_control_metric_map_v03` may enter a formal XD1 result.

Synthetic or historically heterogeneous pools may be used only in the explicitly allowed role recorded in that registry.

## Common representation contract

1. Preserve source order and physical line boundaries where supplied by the source.
2. Preserve page/folio/document identifiers.
3. Use the most diplomatic, unexpanded transcription available for the primary run.
4. Normalize Unicode to NFC only. Do not modernize spelling.
5. Preserve manuscript abbreviation/brevigraph symbols that occur inside tokens.
6. Strip markup and punctuation that are not part of the transcribed token form.
7. Tokenize on transcription whitespace unless the source provides explicit word segmentation.
8. For corpora with paired diplomatic/unexpanded and expanded representations, retain an exact line-level alignment and run both independently.
9. No corpus-specific character mapping may be designed after observing XD1 outcomes.
10. A source with automatic/unverified HTR must pass a separately frozen QC gate before fine-grained character-transition results are admitted.

## Blocking and holdout

- Primary independent block: physical page/folio when available; otherwise the smallest pre-existing manuscript/document unit large enough for the relevant statistic.
- Five deterministic outer folds are assigned at block level using token-count balancing only.
- No linguistic label, writer label, genre label or XD1 outcome may influence fold assignment.
- Hyperparameters are chosen on outer-training material only.
- Sign-flip uncertainty is computed over physical blocks rather than tokens.
- Every reported comparison states the effect and null SD together.
- If `abs(effect)/null_sd < 2`, the lead conclusion is: **the metric does not resolve this contrast**.

## XD1-P1 — within-token memory depth

Question: how much held-out predictive information is added by a second and third preceding character?

For each corpus:
- run left-to-right and reversed-token directions;
- compare order 1→2 and order 2→3 character models;
- smoothing grid: alpha = {4, 16, 64, 256};
- alphabet and lower-order distributions are learned from outer-training material only;
- no section, hand, language or genre conditioning.

Report per condition:
- mean held-out log2 gain per predicted character;
- physical-block sign-flip null SD;
- effect/null-SD ratio;
- positive and negative block counts.

This is the portable counterpart of v03's shallow-form result, not a test for the Voynich-specific hard-zero inventory.

## XD1-P2 — SPACE versus physical LINE_BREAK attenuation

Question: is the dependency from the final character of the previous token to the first character of the next token weaker across a physical line break than across an ordinary within-line space?

Baseline conditions only on:
- boundary type;
- right-token position class (FIRST/MID/LAST);
- right-token length bin (1–2, 3–4, 5–6, 7+).

Enhanced model additionally conditions on the previous token's final character.

No section, hand, writer or content label is used.

Report:
- SPACE edge gain;
- LINE_BREAK edge gain;
- SPACE − LINE_BREAK gain;
- effect and block-level null SD at each frozen alpha.

Only corpora with genuine physical line boundaries are eligible.

## XD1-P3 — previous-token morphology carryover

Question: does coarse shape of the preceding token predict the next token beyond a boundary-aware unigram parent?

Previous-token morphology key is frozen as:
- first character;
- final two characters;
- token length.

Parent:
- global target-token distribution;
- shrunk boundary-specific target-token distribution.

Child:
- adds the frozen previous-token morphology key.

Shrinkage grid:
`{16, 64, 256, 1024, 4096, 16384, 1e9}`.

Lambda is selected using training-only inner folds.

Report held-out gain in bits/token and physical-block sign-flip null SD.

No corpus-specific morphology is introduced.

## XD1-P4 — incremental exact previous-token identity

Question: after XD1-P3's coarse morphology parent is fitted, does the exact identity of the previous token add held-out information?

The exact child is nested strictly above the frozen P3 parent and uses the same training-only shrinkage procedure.

Report:
- exact incremental gain;
- block-level null SD;
- selected exact lambda per fold;
- number of folds selecting the practical off-switch (lambda >= 16384).

No universal closure or universal dependence is inferred from a single corpus.

## XD1-P5 — lagged exact-form recurrence geometry

This is a generic descriptive counterpart to the current-parent v03 R64 theorem. It does **not** reuse the Voynich SG1.1 generator.

Within each physical page/unit, compute exact-token recurrence in four frozen lag bands:
- lag 1;
- lags 2–5;
- lags 6–16;
- lags 17–64.

For each block, compare observed recurrence to a within-block random-permutation null preserving:
- token multiset;
- block length;
- page/unit membership.

Use frozen deterministic seeds and at least 200 permutations/block where computationally feasible.

Report for each band:
- observed minus null recurrence rate;
- null SD;
- effect/null-SD ratio.

The relevant question is whether Voynich's recurrence **shape** is ordinary within a given control class, not which corpus has the smallest Euclidean distance.

## XD1-P6 — abbreviation intervention

Eligible only where the same manuscript lines are available both unexpanded/diplomatic and expanded/regularized.

Run P1–P5 on both representations without changing any settings.

For every metric report:
- unexpanded value;
- expanded value;
- paired difference over identical physical lines/units.

This test estimates how genuine scribal abbreviation itself moves the portable v03 profile.

Augmented or duplicated machine-learning rows are prohibited; only natural manuscript sequence is admissible.

## XD1-A1 — writer/hand stratification (auxiliary)

Where independently supplied writer/hand labels exist, report the XD1 profile by writer and test writer effects only within pre-existing document/book strata that contain crossed support.

A writer effect is **non-identifiable** where writer and document/content regime are not crossed. Absence of crossed support must be reported rather than replaced with a pooled comparison.

This is an auxiliary diagnostic and not part of a global XD1 score.

## XD1-A2 — source-bearing positive control (already executed)

The frozen SG1.1 multi-witness source-control programme remains the qualified positive control for recovery of genuine shared source structure. XD1 does not rerun or reinterpret it.

## Explicit exclusions

XD1 does not use:
- Voynich q/k/t or i-run-specific conditionals;
- the 19 named Voynich hard-zero bigrams as cross-language tests;
- Currier labels;
- Davis hand in the common comparator;
- the killed explicit PAGE_CODEBOOK;
- the old BG42/v11 derivative score;
- old Babuini or nomenclator rankings;
- synthetic JCZS tuned generators;
- any single aggregate "Voynich similarity score".

## First preregistered corpus order

Before external results are inspected, the first formal runs are ordered:

1. Voynich reference under XD1 generic models.
2. Nuremberg Letterbooks 2–5, primary diplomatic/unexpanded representation.
3. Nuremberg Letterbooks 2–5, paired expanded representation for P6.
4. CREMMA Medii Aevi, manuscript-stratified; medical and non-medical witnesses reported separately.
5. Gaskell–Bowern human gibberish, with participant-contamination strata where metadata permit.
6. Bowern et al. algorithmic text transformations.
7. Additional controls only after registry admission.

Nuremberg 6–14 cannot enter the primary quantitative sequence until its automatic transcription passes a frozen QC gate.

## Interpretation rule

XD1 is intended to eliminate or weaken mechanism classes, not identify a historical mechanism by resemblance.

A control that reproduces a Voynich feature demonstrates that the feature is **not diagnostic** of Voynich by itself.

A mechanism class becomes interesting only if it reproduces multiple frozen profile components without target-directed tuning. Even then, historical use requires independent historical evidence.

A failure of a control corpus does not prove that its language, genre or culture is absent from Voynich; it only rejects the tested representation/mechanism as an explanation for the measured surface property.


## Minimum-data gate (frozen before CREMMA outcomes)

To prevent small manuscript witnesses becoming apparent matches through noise:

- P1–P4 require at least 5 physical blocks and at least 1,000 tokens for formal inference.
- P5 requires at least 5 physical blocks and at least 2,000 tokens.
- P6 inherits the relevant metric gate on both paired representations.
- Below-gate outputs may be retained for descriptive audit only and cannot support a comparative conclusion.

This gate was added after CREMMA source ingestion/count inspection but before any CREMMA XD1 scientific outcomes were computed.
