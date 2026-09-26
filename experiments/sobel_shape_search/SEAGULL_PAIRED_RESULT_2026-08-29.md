# Voynich f1r seagull paired-control shape-search result

Date: 2026-08-29
Release: ManuComp `2026-08-26-r15`

## Question

Can a classical Sobel/Canny/chamfer whole-shape matcher use the unusual f1r red `&253` / STA `Yy` glyph (the so-called seagull, including its vertical central squiggle) to retrieve specific manuscript analogues from ManuComp?

## Control

The target was extracted directly from Yale f1r. The negative/near-twin control was the adjacent red `&252` / STA `Yx` form from the same folio, pigment, scale and context, lacking the vertical squiggle. This makes the test specifically sensitive to whether the extra target structure adds retrieval specificity.

## Scanner

Each page was reduced to <=768 px width, converted to Canny edges and distance transform, and searched with 40 target templates: eight widths (`24,32,42,54,70,90,115,145`) x five rotations (`-10,-5,0,5,10` degrees). Lower score is better.

The initial r10 worklist attempt was discarded because its RLS policy exposed zero rows after r15 became the current frozen release. A new scanner was written against the current public `manucomp_release_images` layer and explicitly fails on zero-row scans.

## Clean calibration

Two independent deterministic r15 ID slices were completed with 120 pages each. Every unique page was evaluated against both target and control: 240 unique pages, 480 page-query evaluations, with 40 scale/rotation templates per evaluation. All 480 evaluations completed with zero image errors.

### Slice 8

- target best: **0.56500**
- control best: **0.51151**
- top-80 target/control page overlap: **69 / 80**
- Jaccard overlap: **0.758**
- among all 69 shared candidates, the control scored better on **69 / 69**
- median target-minus-control score on shared pages: **+0.05235**
- top-20 mean score: target **0.58045**, control **0.53246**

### Slice 16

- target best: **0.52454**
- control best: **0.48936**
- top-80 target/control page overlap: **65 / 80**
- Jaccard overlap: **0.684**
- among all 65 shared candidates, the control scored better on **65 / 65**
- median target-minus-control score on shared pages: **+0.06694**
- top-20 mean score: target **0.64409**, control **0.58728**

The same manuscripts and often the same pages dominate both rankings.

## Visual validation

A separate end-to-end focused run regenerated the two clean slices and rendered top-candidate contact sheets. The visible highest-ranking matches are not close structural analogues of the f1r seagull. They are generic illuminated-page structures: miniature frames, historiated/illuminated initials, diagrammatic arcs, margins and dense text/layout edges. The control contact sheet is at least as plausible visually as the target sheet and usually scores better.

## Conclusion

**Reject the whole-shape Sobel/chamfer locator as a viable seagull search method.**

The extra vertical squiggle does not produce measurable retrieval specificity. The simpler adjacent control systematically fits the same candidate pages better, while candidate identity overlaps heavily between target and control. The matcher is therefore responding primarily to generic V/Y-like strokes, frames, curves and page-edge density rather than the defining internal structure of `&253`.

This conclusion is about the retrieval method, not about the historical significance or palaeographic origin of the f1r glyph.

## Next viable method

If the seagull question is pursued further, the search should be component-aware rather than whole-mask chamfer:

1. broad retrieval of V/Y/bird-like candidate regions;
2. explicit verification of a central vertical/s-curved component between two lateral arms;
3. topology/skeleton features (junction count, relative component positions, curvature) or a learned candidate verifier;
4. preserve the adjacent `Yx` glyph as the paired control and require target-vs-control separation before scaling to the full 584,479-image corpus.

A full-corpus run of the current Sobel/chamfer scorer is not justified by these calibration results.
