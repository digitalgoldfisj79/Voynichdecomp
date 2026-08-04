# f116v recto/show-through control programme v0.2

Frozen: 2026-08-04

## Purpose

Resolve whether the residual f116v signal from v0.1 is independent of writing on f116r, using the strongest source-supported control available.

The programme distinguishes three claims:

1. structure explained by f116r show-through;
2. structure not explained by the available f116r control but not validated as writing;
3. recto-independent, cross-family, native-resolution structure compatible with erased f116v writing.

No semantic or generative model may create evidence.

## Acquisition gate

Recursively inventory the public 2014 Lazarus Project / RIT MegaVision archive before analysis.

Required records:

- every raw f116v TIFF path and download identifier;
- every raw f116r TIFF path and download identifier;
- dimensions, bit depth, acquisition family, nominal wavelength, repetition index, hashes and TIFF metadata;
- Yale high-resolution f116r IIIF image and unrelated f115r/f115v images for transform-specificity controls.

If no matching f116r raw cube exists, record `MATCHING_RECTO_CUBE_ABSENT`. A visible-light f116r scan may then be used as a constrained proxy, but the programme may not return `EVIDENCE_PRESENT`.

## f116v cube

Use all raw 16-bit f116v acquisitions. Treat repeated acquisitions as repetitions, not independent spectral bands. Partition by capture family:

- `MB`: reflected multispectral bands;
- `WB`: fluorescence / white-bank captures;
- `TX`: transmitted-light captures.

Register all accepted bands to a frozen reflected-light reference. Preserve transform matrices, ECC scores and exclusions.

## Recto registration

The f116r proxy is aligned through a physically constrained procedure:

1. enumerate the eight dihedral orientation transforms;
2. fit page outline to the f116v page mask;
3. estimate only a small residual translation from transmitted-light structure;
4. select the transform on a frozen correlation score;
5. compare against f115r and f115v unrelated-folio controls.

The correct f116r control must outperform both unrelated folios. Otherwise the recto model is declared non-specific.

## Show-through model

Construct a non-negative, spatially blurred recto-ink basis. For each f116v band, fit the recto contribution on parchment regions that exclude strong f116v front-side ink. The model includes multiple blur scales and a low-frequency parchment term.

Subtract only the predicted non-negative recto contribution. Save per-band coefficients and held-out fit statistics.

## Candidate detector

Learn the spectral signature of surviving f116v front-side ink from strong visible strokes. Candidate residuals must:

- exceed a threshold derived from clean lower-page parchment;
- recur in at least 70% of balanced bootstrap band subsets;
- receive independent support from both MB and WB families;
- survive exclusion of transmitted-light evidence;
- not overlap strong recto prediction or registered recto ink;
- lie inside a frozen inset page mask and outside extreme stain boundaries;
- recur at native resolution in at least two MB and two WB acquisitions.

A page-level candidate additionally requires at least two horizontal line groups. Individual border fragments, holes, creases or isolated components do not qualify.

## Controls

- Correct f116r transform versus f115r and f115v controls.
- Shifted and wrong-orientation recto transforms.
- Synthetic attenuated front-side trace using a real f116v visible-stroke patch.
- Blank parchment false-positive density.
- Bootstrap and leave-family-out stability.
- Native-resolution recurrence.

Thresholds are fixed before inspecting candidate morphology.

## Decision classes

- `NO_RECTO_INDEPENDENT_ERASED_TEXT_SIGNAL`
- `RAW_SIGNAL_EXPLAINED_BY_RECTO_OR_ARTEFACT`
- `RECTO_CONTROL_INCONCLUSIVE`
- `CANDIDATE_RECTO_INDEPENDENT_SIGNAL_MATCHING_CUBE_REQUIRED`
- `EVIDENCE_PRESENT` — allowed only with a matching raw f116r cube and all gates passed.

Physical indentation remains `NOT_IDENTIFIABLE_FROM_AVAILABLE_DATA` unless calibrated multi-direction raking-light images are acquired.

## Evidential discipline

OCR, palaeographic completion, diffusion restoration, super-resolution hallucination and language-model reconstruction are excluded from the verdict. Learned restoration may later be shown only as `VISUALIZATION_ONLY` when every displayed edge has independent acquired-band support.
