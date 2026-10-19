# f116v multispectral recovery and raking programme v0.1

Date frozen: 2026-08-04

## Aim

Use the public 2014 MegaVision f116v image set to answer two distinct questions:

1. Can the surviving marginal text be rendered more legibly without inventing strokes?
2. Is there reproducible evidence of erased, washed, obscured, or impressed writing elsewhere on the page?

The second question is not treated as super-resolution. It is an inverse problem over aligned spectral and, if present, directional-illumination observations.

## Evidential rule

A generative image-restoration output is never evidence by itself. It may be used only as a visualization of signal already supported by non-generative measurements. A candidate hidden stroke must satisfy all applicable gates below.

## Frozen pipeline

### A. Acquisition and provenance

- Download the shared Google Drive folder recursively.
- Preserve source bytes unchanged.
- Record file name, byte size, SHA-256, pixel dimensions, colour mode, EXIF, nominal wavelength or illumination label, and acquisition timestamp.
- Reject exact duplicates and flag probable derivatives by perceptual hash.
- Never overwrite source files.

### B. Geometric normalization

- Detect page orientation and page mask.
- Select the sharpest, highest-coverage visible-light image as reference unless EXIF identifies a canonical reference.
- Register every band using a hierarchy: coarse phase correlation, feature matching, affine ECC refinement, and optional dense RoMa/LoMa refinement on GPU.
- Save forward and inverse transforms plus residual error maps.
- Exclude bands whose registration fails the frozen thresholds.

### C. Spectral cube and non-generative enhancement

All accepted aligned bands are converted to linearized float images and robustly normalized on parchment pixels.

Produce:

- per-band contrast-normalized views;
- median and trimmed-mean composites;
- PCA and minimum-noise-fraction-like components;
- FastICA components;
- non-negative matrix factorization components;
- robust low-rank plus sparse decomposition;
- local spectral-angle and Mahalanobis/RX anomaly maps;
- supervised ink-probability maps trained only on visible surviving strokes and clean parchment controls;
- stability maps across bootstrap band subsets.

### D. Physical-imprint branch

Only run if the inventory identifies multiple illumination directions, raking-light images, or paired directional banks.

Produce:

- opposite-direction difference images;
- gradient consistency maps;
- photometric-stereo normal and relief estimates where illumination geometry is recoverable;
- orientation-reversal tests, requiring a candidate groove/ridge to invert appropriately under opposite illumination.

If the folder contains spectral variation but no directional illumination, the programme must report that indentation recovery is not identifiable from this dataset.

### E. Faithful restoration branch

Run deterministic, non-generative restoration first: denoising, deconvolution, local contrast normalization, and tiled 2x/4x restoration using conservative models.

Optional learned models:

- DocRes / DocWaveDiff / Uni-DocDiff for document restoration;
- HAT or the strongest available restoration-track super-resolution model;
- RoMa or LoMa for cross-band registration;
- DINOv3 features for stroke/background representation;
- SAM 3 only as an annotation assistant, not as a detector of unknown text.

Diffusion or perceptual super-resolution outputs must be labelled `VISUALIZATION_ONLY` and accompanied by the source crop, deterministic restoration, and support map.

## Controls

### Known-text holdout

Mask randomly selected portions of surviving marginal strokes, fit the spectral classifier on the remainder, and test recovery on the hidden portions. Report precision, recall, F1, average precision, and calibration.

### Blank controls

Use page areas judged free of visible writing. The false-positive stroke density in blank controls is a primary gate.

### Band-shuffle controls

Repeat the analysis after spatially permuting or misaligning bands. A genuine cross-band signal should collapse.

### Band-subset stability

A candidate must recur under at least 70% of valid bootstrap band subsets, with median spatial IoU at least 0.50 after skeleton dilation.

### Algorithmic convergence

A candidate region must be supported by at least two independent non-generative families, one spectral and one spatial/topographic when the latter is available.

### Synthetic erasure controls

Create pseudo-erased text from surviving strokes using intensity attenuation, blur, stain transfer, and partial deletion. Tune no thresholds on the real damaged field; thresholds are fixed from synthetic controls and known-text holdout.

## Decision gates

### Gate 1: surviving text enhancement

Pass only if at least one deterministic or spectrally fused output improves blind character-stroke legibility over the best single source band while preserving source-supported edges. Improvement is measured by:

- known-stroke holdout F1 improvement of at least 0.05 absolute; and
- no more than 1% unsupported edge density in blank controls; and
- stable gains across at least three band subsets.

### Gate 2: erased or washed text

A region is a candidate only if:

- non-generative ink probability exceeds the control-derived threshold;
- bootstrap support is at least 0.70;
- the signal survives leave-one-band-family-out analysis;
- the signal is not explained by page texture, crease, stain boundary, JPEG blocks, show-through, or registration residual;
- a stroke-likeness metric exceeds the 99th percentile of blank controls.

A page-level conclusion of `EVIDENCE_PRESENT` additionally requires at least two spatially separated candidate stroke groups with line-like organization or repeated ductus-compatible morphology.

### Gate 3: physical imprint

Pass only if directional-light evidence exists and candidate relief reverses sign under opposite illumination while retaining spatial support. Without directional images, report `NOT_IDENTIFIABLE_FROM_AVAILABLE_DATA`.

## Outputs

- `inventory.csv` and `inventory.json`
- registration audit and excluded-band list
- aligned downsampled cube
- per-method contact sheets
- visible-text enhancement panel
- candidate hidden-signal panel with support overlays
- control metrics and threshold provenance
- machine-readable candidate masks
- `RESULT.md` with one of:
  - `NO_RECOVERABLE_SIGNAL`
  - `VISIBLE_TEXT_ENHANCED_ONLY`
  - `CANDIDATE_ERASED_TEXT_SIGNAL`
  - `EVIDENCE_PRESENT`
  - `NOT_IDENTIFIABLE_FROM_AVAILABLE_DATA`

## Interpretation discipline

The programme detects physical or statistical signal, not language. It does not infer letters from palaeographic expectations, complete words, or use a language model to fill gaps. Any reading of a recovered trace is a separate, blinded palaeographic phase.