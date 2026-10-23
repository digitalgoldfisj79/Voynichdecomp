# Direct image arbitration and crop-contamination control (2026-10-23)

**Status: direct subset review complete, preregistered Stage 1 QC STILL FAILED.** The fourth historical-mechanism hypotheses remain untested; no confirmatory Stage 2 was run.

The original Yale IIIF images for **all 15 f70v1 dark-Aries figures** were inspected directly in individually indexed crops and contact sheets. Some star-tail judgements remain uncertain rather than being forced. Particularly:
- #1: fine attached connector visible, consistent with Stolfi;
- #2: no discernible separate star stem, consistent with Stolfi;
- #3: star ray and fingers overlap; separate stem genuinely uncertain;
- #4: star next to fingertips, separate stem absent; **both models said yes but Stolfi said no**;
- #8: long thin connector visible;
- #7, #9 and #13: not reliably separable at displayed image resolution.

A contamination error surfaced on dark Taurus: whole-crop red-pixel counts on figures #26, #28, #29 and #30 include **painted material from a neighbouring foldout diagram**, rather than the nymph being measured. **Full-crop red-pigment attribution is retracted.** A new pixel measurement on a central ROI (x 22–78%, y 17–69% within rotated IIIF crop) with threshold R>G+36, R>B+25, R>95, G<150, B<150 yields:

| Panel | Figures with ≥100 central red pixels | Total |
|---|---:|---:|
| Light Taurus f71v | 12 | 15 |
| Dark Taurus f72r1 | 3 | 15 |

This difference is visually convincing as a colour/garment contrast, but **not independent of the garment feature used for Stage 0's discovery**. Figure-level Fisher p=.002814 is merely descriptive: nymphs aren't 30 independent production sessions, the threshold and ROI were chosen post hoc and the detector is not independently ground-truthed. **Do not move Stage 2 gate or use this as proof of a new exemplar, scribal session, or retoucher.**

The two original machine passes remain archived in `STAGE1_RAW_TWO_MODEL.json`. Original figure/IIIF crosswalk is Supabase `voynich_zodiac_mask_catalogue_crosswalk_20261023`. Pixel runs were Hugging Face CPU jobs `6adb9b581e112550ce7d6ec2` and `6adb9b641e112550ce7d6ec8`.

See companion downloaded artifact `Voynich_Zodiac_Visual_Tiebreak_20261023.zip` for a 15-row manual figure disposition and all 30 pixel values; it is not a verified historical decoding.
