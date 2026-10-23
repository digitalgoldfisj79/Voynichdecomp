# Original Yale image tie-break — figure red-pigment measurement, 23 October 2026

**Status: source-pixel exploratory verification completed on all 60 preregistered Aries/Taurus nymph positions. This does NOT pass or supersede the frozen Stage-1 inter-rater agreement gate. Confirmatory Stage 2 NOT RUN.**

A contact sheet containing 15 individual original Yale crop previews for each of the four half-sign panels was directly inspected. Two-model disagreement on star tails (including dark Aries #4) remains an observational problem; Stolfi's per-figure table is the external source control.

Independent, deterministic RGB area measurements were computed for each of the 60 registered **nymph bounding boxes**, not the larger figure-plus-star-and-label crops. Photo URLs and all 60 numerical readings: [STAGE1_PHOTOMETRY_60.json](./STAGE1_PHOTOMETRY_60.json).

| Panel | Red-pixel area ≥0.001 of paper pixels | ≥0.002 | Strict red ≥0.001 |
|---|---:|---:|---:|
| Dark Aries, f70v1 | **0/15** | 0/15 | 0/15 |
| Light Aries, f71r | **12/15** | 11/15 | 9/15 |
| Light Taurus, f71v | **11/15** | 11/15 | 4/15 |
| Dark Taurus, f72r1 | **0/15** | 0/15 | 0/15 |

At red threshold 0.0003 the counts are 2/15, 12/15, 13/15, 0/15 respectively. At 0.005 they are 0/15,10/15,6/15,0/15. Nine transient Yale IIIF HTTP500 bbox subregion failures were recovered through expanded cropped requests. All 60 numeric results are present.

### Materially revised interpretation
**The raw red-paint pattern is LOW–HIGH–HIGH–LOW in zodiac page order**, dark Aries → light Aries → light Taurus → dark Taurus. This is REPEATED across two signs and at least two bifolia. Light and dark Taurus are part of the same opening and digitised exposure, so a general page-exposure confound is weakened.

The earlier Stage-0 k=2 cut after light Taurus is NOT uniquely diagnostic of a one-time exemplar or production-session switch. The non-monotone contrast tracks the light/dark *variant treatment*. It is compatible with a planned iconographic or painting programme **and** with targeted secondary painting; historic drawing/painting order is unestablished and the cause remains unidentified.

This differentiation was NOT newly discovered in this run: older Voynich scholars (Stolfi, Zandbergen, Petersen and others) noticed the colour and costume contrast. New contribution is a complete *figure-bound* source-pixel measurement, independent of error-prone vision-model attribute names.

Stolfi records about nine figures dressed on dark Aries even though there is effectively zero strong red torso pigment there: **dress shape is not the same measured variable as red colouring**.

### Statistical and epistemic cautions
This is an **exploratory alternative measurement developed after observing and debugging the failing two-model instrument**, not a blinded preregistered Stage-2 feature. It is related to the original selection variable 'clothing', so it cannot independently confirm the Taurus boundary. The correct independent unit for two light/dark contrasts is *two paired signs*, NOT 60 independently produced artefacts. A sign-preserving light/dark swap test has only 4 allocations (minimal one-sided p=0.25) — thus the historical mechanism is not statistically identified.

Method: mask includes pixels R>85,G>65,B>50, mean RGB>105. 'Red' R>105 and R>1.42G, R>1.27B and G>43; 'strict red' R>120, R>1.57G and R>1.39B. Pixel ratio to eligible mask pixels, count ≥0.001. Photo pixels do not establish the age or composition of the paint. Figure boxes include some parchment and may exclude extensions/retouched portions. Threshold and feature selection are POST HOC.

### Quality gate / next discriminator
The two machine readers' Stage-1 QC continues to FAIL (star-tail κ=.40, red pigment agreement κ=.58, gesture κ=.43, star-ray within-one 50/60); third-viewer crop inspection does not replace independent expert agreement. The next source-level discriminator is *stroke/pigment chronology and physical overlap*, not another statistical search for a red-paint breakpoint. Multispectral/infrared images and layer-aware inspection would be needed before asserting earlier vs later painting.

Primary sources: [Yale Quire 11](https://voynich.nu/q11/index.html); [Stolfi's dark Taurus](https://vib.tamagothi.de/index.php?id=f72r1&show=page). Relevant prior observations: [Voynich Portal zodiac](https://voynichportal.com/tag/zodiac/).

**Retain both potential mechanisms: designed light/dark variants and selective later colouring. Neither is proved.**
