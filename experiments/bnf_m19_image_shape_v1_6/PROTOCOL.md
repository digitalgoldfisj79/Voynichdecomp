# BnF M19 Image Shape v1.6 — Frozen Protocol

Date: 2026-08-09
Parent: `experiment/bnf-m19-image-continuous-v1.5-20260809` at `6582d0631cec8df64e1a8df8203bac7d3d8b889f`.
Image repository: `Digitalgoldfish79/vdino3-crops`, revision `ea597db8ff2c06631c4c311d90c8cf0418f5e26c`.
Seed namespace: `M19IMAGEv16`.

## Motivation

v1.2–v1.4 recovered reproducible image-derived boundaries but not stable discrete 19-state identities. v1.5 showed that unconstrained continuous DINO emissions are not identifiable even on synthetic M19 positive controls. v1.6 therefore asks whether **independently observable binary-shape/topology features** can stabilize a surface-state inventory before any language model is consulted.

No EVA/transcription field may be used for fitting, model selection, feature engineering or state assignment. `word`, `eva_aligned`, `word_len` and related text fields remain sealed until terminal audit.

## Stage S0 — retained-crop feasibility screen (binding prerequisite)

The private HF repository retains PNG crops for one 8-folio shard only (`crops/crop_shard_000`), containing 12,990 `ccmerge/norm` image units. This stage is a feasibility screen only; it may authorize full-corpus crop regeneration but may not produce a language result.

1. Sort the eight folios by SHA-256 of `M19IMAGEv16shape::<folio>`; first four are ShapeTrain, last four ShapeTest.
2. Within each folio retain at most 750 `ccmerge/norm`, `low_conf=false` crops by smallest SHA-256 of `M19IMAGEv16unit::<id>`. This caps the screen at 6,000 images and prevents runtime-driven post-hoc sampling.
3. Convert every crop to an ink-on-black 48x48 normalized canvas, preserving aspect ratio.
4. Compute the following feature blocks prospectively:
   - **T** topology/geometry: log aspect ratio, ink fraction, bounding-box extent, perimeter/area, solidity, eccentricity, Euler number, hole count, connected-component count, skeleton length/area, endpoint count, junction count, horizontal/vertical centroid, horizontal/vertical second moment (16 values; degenerate values clipped).
   - **H** HOG: 9 orientations, 8x8 pixels/cell, 2x2 cells/block on the 48x48 canvas, standardized on ShapeTrain then PCA-whitened to min(32, n_features, n_train-1) dimensions.
   - **R** raster: downsample to 24x24, flatten, standardize on ShapeTrain then PCA-whiten to 32 dimensions.
   - **HT** concatenated H + standardized T, then L2-normalize.
   - **RT** concatenated R + standardized T, then L2-normalize.
   - **HRT** concatenated H + R + standardized T, then PCA-whiten to 48 dimensions and L2-normalize.
5. For each representation in {T,H,R,HT,RT,HRT} and K in {19,25,31,38}, fit MiniBatchKMeans independently with seeds 408 and 409 on ShapeTrain.
6. Match centroids by Hungarian maximum cosine. On ShapeTest report occurrence-weighted assignment stability and cosine silhouette (deterministic max 3,000-image silhouette sample).
7. A representation/K passes S0 only if:
   - cross-seed ShapeTest assignment stability >=0.75;
   - every matched cluster has >=20 ShapeTest members;
   - every cluster occurs in >=2 of the 4 ShapeTest folios;
   - cosine silhouette >=0.08.
8. If multiple pass, select maximum stability; ties within 0.01 choose higher silhouette, then smaller K, then simpler representation in order T,H,R,HT,RT,HRT.
9. If none pass, verdict is `SHAPE FEATURES DO NOT STABILIZE M19 SURFACE STATES`; do **not** regenerate full-corpus PNG crops and do not run a language test.

## Stage S1 — full-corpus regeneration (only if S0 passes)

If S0 passes, regenerate only the selected image unit type/features for the already frozen v1.2 T12/H12/C12 folio split from Yale/IIIF using the recovered crop pipeline. Do not store or inspect EVA labels. Fit the selected representation/K on T12 only.

The full-corpus visual gate repeats the v1.2 criteria: T12 internal cross-seed stability >=0.75; H12 frozen-centroid mapped coverage >=0.90; every cluster recurs in >=5 H12 folios and has >=50 H12 members.

## Stage S2 — M19 positive controls (only if S1 passes)

Use the exact v0.9 BnF M19 emission law and six fresh controls (Latin, Italian, German, French, Arabic, Spanish), 45k train / 39k hold. Synthetic image features are generated from the empirically fitted cluster-feature distributions with folio-level nuisance sampled from T12 residuals. The state solver must recover:

- 6/6 correct language rank;
- minimum correct-language margin >=0.05 nats/unit;
- median numerical-state recovery >=0.95;
- minimum recovery >=0.85;
- minimum independent-fit agreement >=0.90.

Failure -> `SHAPE-M19 INSTRUMENT NOT QUALIFIED`; Voynich language scoring prohibited.

## Stage S3 — Voynich language gate (only if S2 passes)

Fit separate T12 legal M19 maps under the frozen eight-language panel. Two independent fits each. H12 candidate requires rank 1, >=0.05 nats/unit margin, >=0.90 fit agreement and >=0.90 mapped coverage. Only a passing H12 candidate unlocks the fixed-map C12 confirmation. C12 requires rank 1, >=0.05 margin and positive candidate-v-runner-up margin in all four deterministic C12 buckets.

No decoded strings may be inspected before C12 confirmation.

## Verdict vocabulary

- `SHAPE FEATURES DO NOT STABILIZE M19 SURFACE STATES`
- `SHAPE-M19 INSTRUMENT NOT QUALIFIED`
- `NO SHAPE-M19 LANGUAGE SIGNAL`
- `H12 SHAPE-M19 CANDIDATE / C12 FAILED`
- `CONFIRMED SHAPE-M19 SIGNAL <language>`

No threshold may be relaxed after observing a result.
