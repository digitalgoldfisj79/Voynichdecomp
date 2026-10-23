# Stage 1 executed — QC instrument failure (23 October 2026)

**Decision:** **QC_FAILED_INSTRUMENT. STOP before confirmatory Stage 2.** The initial Taurus-boundary hypothesis has **not** been tested on reliable independent S/I/L attributes. This finding is **not** evidence against planned iconography, exemplar switch, production session changes or retouching.

## Executed against original Yale scans
- Original *frozen* four half-sign panels: dark Aries f70v1, light Aries f71r, light Taurus f71v, dark Taurus f72r1; **60 figure records** (15 each).
- Replaced full-page placeholders for 45 figures with proposed nymph-star crop URLs from the archived crosswalk (not necessarily independently validated).
- Two different multimodal machines, GPT-4.1 and Gemini 2.5 Flash, each supplied independent image-only attribute prompts without labels, signs or hypotheses; **120/120 machine readings were saved**.
- Original raw model answers and crop evidence links: [STAGE1_RAW_TWO_MODEL.json](./STAGE1_RAW_TWO_MODEL.json).
- This is a **two-model instrument trial**, *not* two expert palaeographers conducting independent blind readings; passing the machine reliability metrics alone would not satisfy all physical provenance requirements.

## Registered coverage and reliability gate
Each field was only scored when both models supplied an observable category. Required **≥48/60 comparable** plus **Cohen's κ ≥0.70**, except star-point count ≥48 and ≥85% agreement within one ray.

| Measured field | Comparable / 60 | Raw agreement | κ | Gate |
|---|---:|---:|---:|---|
| Star tail / attached connector | 60 | 70.0% | **0.400** | FAIL |
| Star-outline type | 60 | 100% | undefined (constant) | FAIL / no discrimination |
| Star colour | 60 | 76.7% | **−0.058** | FAIL |
| Red pigment on figure | 60 | 78.3% | **0.577** | FAIL |
| Blue pigment on figure | 60 | 83.3% | **0.504** | FAIL |
| Double outline | 60 | 91.7% | **0.000** | FAIL / high imbalance |
| Retouch candidate | 0 | not measurable | undefined | FAIL |
| Label stroke variation | 60 | 61.7% | **0.266** | FAIL |
| Label spacing | 60 | 100% | undefined (constant) | FAIL / no discrimination |
| Gesture (secondary) | 60 | 78.3% | **0.433** | FAIL |

**Star-ray counts:** Both models recorded numeric values for all 60. Exact count agreement **43/60=71.7%**, within-one agreement **50/60=83.3%**, below 85% gate. GPT-4.1 returned eight points for **53/60**, Gemini 2.5 Flash for **45/60**; this concentration suggests systematic defaulting to eight rather than robust independent counting.

**Source crop identity:** Models marked a figure visible in both **55/60 (91.7%)**, below the preregistered ≥95% identity gate *even as a proxy*. Four crosswalk angular discrepancies exceed ten degrees: f70v1 #13 (10.7°); f71v #10 (10.8°), #12 (13.6°) and #14 (22.8°). Only **51/60 (85%)** satisfy both "figure present" and the simple ≤10° proximity check; proximity is not itself proof of intended source identity.

## Required external adjudication: Stolfi, dark Aries
Read the original Stolfi per-figure descriptive table, not a model-generated transcript, for **15 positions of f70v1**:
- Tabulation contains **11 star tails present, 4 absent**, when mapped to clockwise outer positions 1..10 then inner positions 11..15.
- GPT-4.1 vs Stolfi tail coding: **10/15 agreement**, κ=**−0.119** (overcalled tail yes).
- Gemini 2.5 Flash vs Stolfi: **12/15 agreement**, κ=**0.328** (also overcalled tail yes).
- The external source itself is observational, not physical proof of manuscript ink chronology; position mapping and any damaged areas still require an expert crop check.
- This source adjudication does **not** repair the models' full dataset or licence choosing whichever model agrees with the expected regime.

## Historical inference
- **Stage 0:** exploratory best cutpoint f71v → f72r1 from clothing / containers / headwear. Explained variation **≈84.8%**, maximally searched exact p=**0.07143**. Not confirmed.
- **Stage 1:** **QC_FAILED_INSTRUMENT**; the proposed independent fingerprint (star drawing, ink properties, label execution) cannot be tested at registered quality.
- **Stage 2:** **NOT RUN** because Stage 1 failed. Do not use the 120 model predictions to generate another post-hoc boundary p-value or classify a copying session.
- **Stage 3/4:** conditional, **NOT RUN**.
- **No conclusion** on the four competing historical mechanisms. Blank or contradictory annotation means measurement failure, **not absence** of meaningful content.

## Conditions for resumption (not automatically initiated)
Human/expert double annotation on registered 60 positions, with nymph and star crop identity verified against full-page Yale coordinates; blinded to the proposed breakpoint where practical; Stolfi notes withheld until adjudication; star ray and tail and pigment/ink notes verified; re-run registered QC unchanged. The choice to remeasure must not move the discovery breakpoint or tune the null test after observing the models' output.

## Persistent records
Supabase project `ymaqlcfjmdwncdbjprmw`:
- `voynich_zodiac_stage1_trial_f70v1_20261023`
- `voynich_zodiac_stage1_trial_f71r_20261023`
- `voynich_zodiac_stage1_trial_f71v_20261023`
- `voynich_zodiac_stage1_trial_f72r1_20261023`
- `voynich_zodiac_stage1_qc_trial_20261023`

**Methodological policy:** Distinguish apparatus/instrument failure from negative historical hypothesis. No 60 independent production events: the statistical unit for later causal comparison remains physical panels/bifolia, not isolated figures.
