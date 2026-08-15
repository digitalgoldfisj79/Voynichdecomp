# Structured Production Specificity Audit v0.1 — 2026-08-15

## Status
Post-hoc mechanism-specificity audit. This does **not** alter the frozen primary verdict of Medieval Magic Formula Discriminator v0.2 (`NO_ROBUST_MAGIC_AFFINITY`). It addresses the secondary finding that Voynich has representation-robust local F2/F3 family/near-copy geometry closer to productive voces class C than ordinary prose A.

## Why this audit exists
The wider Voynich programme already supplies strong alternative mechanism evidence that must not be ignored:

- the best current descriptive/generative account is section-conditioned and slot-like rather than a simple running-language or cipher model;
- Currier/section structure emerges under production modelling;
- prior generator work showed that constrained copy/mutate/template mechanisms can recover a substantial fraction of Voynich distributional behaviour, with Gen-SP the best non-transcription generator in the existing hierarchy;
- broad historical cipher families and several notation/cipher alternatives have repeatedly failed decisive held-out gates;
- the current medieval-magic audit finds long-range/compression behaviour ordinary-prose-like in 3/4 representations but local family/near-copy geometry C-like.

Therefore the relevant competitor is no longer merely “ordinary prose”. It is **non-magical structured production**. This audit asks whether C is specifically needed to explain the local geometry.

## Frozen post-hoc target metrics
The primary local subspace is exactly the six persistent metrics identified before this run:

1. `F2_oneedit_component_frac`
2. `F2_oneedit_degree`
3. `F2_shared_core_ratio`
4. `F2_tok_len_mean`
5. `F2_tok_len_std`
6. `F3_nearcopy_lag10`

The extended local space adds qualified `F3_mutation_advantage`, `F4_init_final_jsd`, and `F4_line_medial_len` when available. No metric is added or removed after the control results are seen.

## Non-magical controls
All synthetic controls are trained **only on external class-A training texts**, never on Voynich and never on class C:

- `NM_MARKOV`: ordinary word-level Markov production;
- `NM_SLOT`: A-derived prefix/core/suffix slot assembly;
- `NM_COPYEDIT`: recent-token copy with single-character mutation;
- `NM_INFLECT`: A-derived productive stem/suffix families;
- `NM_RECIPE_TEMPLATE`: formulaic technical/medical/cooking template production from the non-charm recipe sources;
- `NM_DEVOTIONAL_TEMPLATE`: formulaic non-magical religious/template production;
- `NM_ABBREV`: ordinary A text with abbreviation/fusion;
- `NM_PROFILE_SWITCH`: source-conditioned copy/edit chunks, testing section-conditioned production without target leakage;
- `NM_COPY_ABBREV`: copy/mutate plus abbreviation;
- `REAL_RECIPE` and `REAL_DEVOTIONAL`: the actual A-training windows in those subgenres, retained as small historical reality controls.

The synthetic controls are generated at a target-independent token grid of 20, 40, 80, 160, 320, and 640 tokens, 50 samples per grid point. Voynich token count is used **only after generation** to select length-matched control samples.

## Representations
Score independently in the already frozen hierarchy:

1. `RF_WORD`
2. `STA_FAMILY`
3. `FULL_STA`
4. `AAA_CONNECTED`

No one-representation result is promoted.

## Calibration and scoring
External A/C held-out samples define the metric standardisation. For each Voynich folio, each non-magical control class is represented by a length-matched median centroid (±25% token count; if fewer than 30 samples, nearest 30 in log-token length). Define:

`margin_C_specific = d(best non-magical) - d(C)`.

Positive means class C is closer than every non-magical structured control; negative means at least one non-magical structured mechanism is closer.

Inference is based on paired folio margins and bootstrap median CIs. There is no tuned absolute distance threshold.

## Cross-layer decision
- `LOCAL_MAGIC_SPECIFICITY_SUPPORTED`: positive median in >=3/4 representations and bootstrap lower bound >0 in >=2/4.
- `LOCAL_SIGNAL_GENERIC_STRUCTURED_PRODUCTION`: negative median in >=3/4 and bootstrap upper bound <0 in >=2/4.
- otherwise `LOCAL_SPECIFICITY_NONRESOLVING`.

These are frozen before the control run. They are a post-hoc specificity decision rule, not a preregistered primary hypothesis test.

## Section audit
Report section-level median margins and fraction of folios for which C beats all non-magical controls. Herbal-A/Herbal-B are of particular interest because they retained a residual C-like signal after length adjustment, but no section receives special fitting or thresholds.

## Scientific firewall
A failure of C specificity means only that the local F2/F3 resemblance is generic to structured production. It does not refute the independent f116v marginal charm evidence. Conversely, a positive result does not establish that the main manuscript is magical; it only makes productive magical formulae a more specific structural comparator for the local family geometry.
