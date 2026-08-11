# Q9/Q10 Hybrid Scale Comparanda Programme v0.4

Date: 2026-08-11
Branch: `experiment/q9q10-hybrid-comparanda-v0.4-20260811`
Primary HF job: `6a7ab23a3b2516b29b154d04`

## Executive result

**PASS under the frozen astronomical-only calibration.**

v0.4 combines DINOv3 candidate generation with explicit structural primitives used as a family-specific shortlist reranker rather than as one global search metric.

The method passed all three promotion gates:

1. DINO astro retrieval reproduced: median top-20 enrichment **2.6602x**, **7/8** held-out manuscript queries above baseline.
2. Structural extraction quality: **206/208** valid primitive vectors, **2 failures = 0.96%**, below the <=10% ceiling.
3. Hybrid reranking improved held-out performance: control-only alpha **0.55 DINO / 0.45 structure**; median top-10 enrichment rose from **3.2741x DINO-only** to **3.6833x hybrid**, with **6/8** queries above baseline.

This is the first scale retrieval method in the Q9/Q10 programme to satisfy the corrected astronomical calibration while demonstrating incremental value from the structural layer.

## Corpus and coverage

Frozen database universe: **1,340** dated records, 1250–1500.

- resolved unique image URLs: **500**
- usable downloaded visual entities: **469**
- manifest unavailable: **646** rows
- folio unresolved: **141** rows
- image download failures: **31**

The method is validated on the accessible corpus, not on an exhaustive census of medieval astronomical diagrams.

## Architecture

1. DINOv3 grayscale/edge retrieval produces manuscript-diverse top-30 shortlists.
2. Qwen2.5-VL-3B extracts constrained diagram primitives only for target/control shortlists.
3. Voynich targets use frozen families: annular/concentric, radial/compartment, star-field/heavenly, corner/quadrant, or figure-bearing.
4. Family-specific structural scoring reranks the DINO shortlist; hard contradictions receive fixed penalties.
5. The DINO/structure coefficient is selected only on eight held-out `astro_diagram` controls.
6. Target rankings are generated once after alpha is fixed.

No target ranking participates in calibration or alpha choice.

## Pre-audit machine leads

- **f69r -> Leipzig UB Ms 1483 f.2r (1442):** hybrid rank 1 from DINO rank 10, structural score 0.5356. This is the clearest demonstration of the structural layer rescuing a relevant astronomical-family candidate.
- **f70r1 -> Leipzig UB Ms 1483 f.2r:** hybrid rank 3 from DINO rank 10, structural score 0.6450.
- **f70r1 -> Lyon BM Ms 172 f.29:** hybrid rank 1, structural score 0.5653.
- **f68r3 -> Munich BSB Cgm 120 f.1v / Lyon BM Ms 172 f.29:** hybrid ranks 1/2.
- **f68v2 -> Lyon BM Ms 172 f.29 / Cgm 120 f.1v:** hybrid ranks 1/2.
- **f67r1 -> Lyon BM Ms 172 f.29 / Cgm 120 f.1v:** hybrid ranks 1/2, structural score 0.8966 each.

These were retrieval leads only and were subjected to exact-folio audit.

## Strong machine-level negative signals

- **f67v2:** every leading candidate fails the target-defining corner-topology requirement; no candidate reproduces the four-corner linked network.
- **f68v3:** every leading candidate fails the essential T-O-centre/compound morphology; no candidate replaces the stronger source-led comparator family.
- **f69v:** all leading candidates fail the 28-fold count/topology requirement; no candidate reproduces the combined 28-member pipe-ring signature.

## Post-unblinding exact-folio audit

A separate post-ranking audit used `Qwen/Qwen2.5-VL-7B-Instruct` on the exact Voynich target crop and exact candidate folio image. Fourteen of the strongest or most diagnostic target-candidate pairs were audited.

**Result: 0 Tier A, 0 Tier B, 14 Tier C generic-family neighbours.**

Representative outcomes:

- **Lyon BM Ms 172 f.29** repeatedly ranks highly for radial/annular targets but is only a generic circular astronomical neighbour on exact inspection.
- **Munich BSB Cgm 120 f.1v** is a useful radial/calendar family neighbour but not a close match to f67r1, f68r3 or f68v2.
- **Leipzig UB Ms 1483 f.2r** demonstrates that the structural reranker works, but exact inspection still reduces the f69r and f70r1 comparisons to generic-family level because centre type, counts, colour-coded axes and perimeter structures differ.
- **f67v2 vs Abbeville Ms 12 f.4v:** the candidate lacks the four linked corner faces/branching network.
- **f68v3 vs Lyon Ms 172 f.20v:** the candidate lacks the T-O-like partition, starry annulus, nebuly boundary and eight spiralling text bands.
- **f69v vs Leipzig Ms 1164 f.70v:** the candidate does not reproduce the 28-member pipe/tube ring or 28-fold count.

Therefore **no new Tier A or Tier B external comparator is promoted from v0.4**.

## Final interpretation

Two propositions are now separated cleanly:

1. **Retrieval proposition — PASS.** DINO plus family-specific structural primitives improves held-out astronomical-family retrieval over DINO alone.
2. **Historical-comparator proposition — no new close hit.** The strongest retrieved folios remain generic family neighbours after exact-image inspection.

The scale search therefore strengthens the negative evidence around the most distinctive Q9/Q10 constructions rather than displacing the existing source-led comparanda.

## Compute closeout

The first v0.4 attempt was discarded after a right-padding warning in decoder-only batched Qwen generation. The corrected run changed only tokenizer padding to `left`; no scientific parameter changed. The corrected primary run completed normally in ~15.4 minutes. The post-unblinding audit also completed. **No Hugging Face jobs remain running.**
