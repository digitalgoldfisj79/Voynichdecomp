# Full-manuscript blind direct-pixel boundary replication v0.1

Frozen: 2026-08-20, before full-manuscript label reveal.

## Question

Does the direct ink-to-ink boundary-gap effect reported by Rozanova & Temerev (arXiv:2608.17096v1) generalise from their six-folio blind audit to the full eligible Voynich manuscript?

The directional estimand is **mean gap(certain) - mean gap(uncertain)**. The published six-folio primary estimator reported +1.842 px at 700-pixel page width (265 certain, 21 uncertain after blind QC).

## Source layers

- ZL IVTFF 2b: `https://www.voynich.nu/data/previous/ZL_ivtff_2b.txt`; `.` = certain word space, `,` = uncertain word space.
- Voynichese.com runtime word rectangles: `https://www.voynichese.com/1/data/folio/script/{folio}.js`, in the matching 636x900 legacy viewport.
- Yale MS 408 IIIF Presentation 3 manifest: `https://collections.library.yale.edu/manifests/2002046`.
- Authors' primary direct-pixel algorithm: `lrozanova/voynich-units`, `analysis/direct_pixel/measure_direct_pixels.py`, blob SHA `28e8fd46a0212d31c5b607661159303632b81d47`.

## Eligibility and alignment

1. Running paragraph (`P`) loci only.
2. Only explicit intra-line `.` and `,` separators; drawing interruptions are excluded.
3. Voynichese boxes are registered to a 2500-pixel-wide Yale derivative using SIFT + CLAHE + USAC_MAGSAC; no proportional coordinate scaling replaces registration.
4. ZL and Voynichese token sequences are aligned per folio with exact matching blocks. A boundary is eligible only when both flanking tokens occupy consecutive positions in the same exact matching block.
5. Label-blind geometry gate: the right box must lie to the right and the two box centres must be within 0.80 times their summed heights vertically.
6. Registration gate frozen before outcome: >=12 inliers, inlier ratio >=0.35, median reprojection error <=4.0 px at registration scale.

All exclusions and alignment denominators are reported, including the uncertain-boundary retention rate.

## Blinding

The alignment stage writes two artifacts:

- blind manifest: anonymous ID, folio/line, IIIF service, locator geometry and registration diagnostics only;
- sealed key: anonymous ID plus `.`/`,` label, token identities, hand and quire.

The measurement job downloads **only the blind artifact**. Its input schema is mechanically rejected if it contains label/group/certain/uncertain or token-identity fields. The sealed key is first downloaded by the analysis job after blind measurement CSVs and hashes have been emitted.

## Primary measurement

700-pixel-wide Yale IIIF derivatives. The morphology estimator is algorithmically identical to Rozanova & Temerev's primary `measure_direct_pixels.py`: bounded grayscale Otsu + HSV saturation mask, connected-component cleanup, 8-92% vertical support, midpoint split search, and direct nearest-ink gap. Page caching is an execution optimisation only.

Primary inclusion is frozen as `qc == ok` and finite gap. `review_far_edge` and other QC categories are excluded without labels. An all-finite sensitivity is also reported.

## Resolution extension

A separate 2500-pixel arm is frozen before unblinding. Locator geometry comes from the same registration. Pixel-distance constants are multiplied by actual image width / 700 and component-area cutoffs by its square. Results are reported both in raw pixels and rescaled to 700-pixel-equivalent units. This arm is an extension, not part of the exact 700-pixel replication.

## Threshold sensitivity

At 700 px, run threshold offsets -30,-25,...,+30 exactly as the authors' published sensitivity grid. The predeclared central robustness band is -15 through +15; all seven effects must retain positive direction for the `threshold_core_all_positive` flag.

## Inference

Primary:

- pooled certain-minus-uncertain difference;
- Cohen d;
- folio-cluster bootstrap (10,000 resamples);
- within-line label permutation preserving each line's group counts (50,000 replicates); report null mean, **null SD**, effect/null-SD ratio and one/two-sided p-values;
- exact sign test across informative folios.

If |effect| / null SD < 2, reporting must lead: **the metric does not resolve this**.

Secondary, predeclared descriptive strata: hand, quire and canonical section where each stratum has >=50 certain and >=10 uncertain eligible measurements. No stratum may replace the primary pooled adjudication.

## Frozen adjudication

`REPLICATES_PRIMARY` iff all are true:

1. 700px primary certain-minus-uncertain difference > 0;
2. within-line two-sided permutation p < 0.05;
3. folio-block bootstrap 95% interval excludes zero on the positive side.

`REPLICATES_AND_SCALE_STABLE` additionally requires:

4. 2500px effect, expressed in 700px-equivalent units, > 0;
5. every threshold effect from -15 through +15 > 0.

Otherwise results retain their measured values and are reported under the corresponding failure/sensitivity status; no thresholds are retuned after unblinding.

## Caveat fixed in advance

Voynichese locator boxes and the authors' locator boxes share a human segmentation tradition. Therefore box-to-box distance is not treated as independent evidence. The load-bearing measurement is ink-to-ink distance extracted from Yale pixels inside those locators.
