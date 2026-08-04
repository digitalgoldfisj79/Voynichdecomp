# f116v palaeographic extraction programme v0.1

Date frozen: 2026-08-04

## Objective

Extract an uncertainty-preserving glyph apparatus for the surviving marginal writing on Beinecke MS 408 f116v. The programme targets physical stroke support, segmentation, and ranked character hypotheses. It does not perform language decipherment or complete plausible words.

## Evidence hierarchy

1. Acquired pixels and cross-view stroke persistence.
2. Non-generative segmentation and repeated-form morphology.
3. Domain-specific HTR hypotheses.
4. Generic OCR controls.

A model prediction without acquired-pixel support is `MODEL_ONLY` and cannot enter the transcription.

## Data

Use only the f116v true-colour TIFF, expert multispectral composites, and selected raw 16-bit f116v bands already inventoried in stage 2. Record Drive file ID, source path, SHA-256, dimensions, bit depth, and derivative status.

Primary pilot views:

- `Lab_true_color_TIFF/Voynich_116v_PSC.tif`
- `Processed_Images/Voynich_116v/Voynich_116v_bands01-22_RF+FL_cal_faded_text_RB+UV_FL_8bands_PCA_R-1G-2B3_hue+20b_r90_BW.tif`
- `Processed_Images/Voynich_116v/Voynich_116v_bands01-22_RF+FL_cal_faded_text_RB+UV_FL_8bands_PCA_R1G2B3.tif`

## Models

Primary HTR:

- Kraken with CATMuS Medieval, latest verified Zenodo release. Run without dictionary correction or external language model. Preserve raw character confidences and geometry where available.

Independent HTR controls:

- `medieval-data/trocr-medieval-base`
- `medieval-data/trocr-medieval-cursiva`
- `medieval-data/trocr-medieval-humanistica`
- `medieval-data/trocr-medieval-semitextualis`
- `medieval-data/trocr-medieval-textualis`
- `Teklia/pylaia-himanis` where operational, without its external language model.

Generic hostile controls may include Fal Florence-2 OCR and GOT-OCR 2.0. Their outputs never determine the accepted reading.

## Stage 1: one-line pilot

1. Derive line crops without OCR using projection profiles and connected components on the rotated monochrome expert composite.
2. Register matching crops from true-colour and second PCA view.
3. Produce deterministic binarisation and source-support masks.
4. Run CATMuS and at least two independent TrOCR models on each view.
5. Run synthetic fading and low-resolution controls using acquired visible strokes.
6. Scale to all lines only if the pilot shows non-trivial cross-view or cross-model stability above blank and degradation controls.

## Stage 2: full four-line extraction

For every line:

- retain original, contrast-normalised, and binarised crops;
- generate connected-component, projection, contour/watershed, and CTC-derived segmentation hypotheses;
- run all operational HTR models independently;
- compute image-to-image and output-string stability across views;
- create glyph embeddings and unsupervised repeated-form clusters;
- attach every character hypothesis to an acquired-pixel support mask.

## Status vocabulary

- `SUPPORTED`: stable acquired geometry across at least three independent views or view families and compatible hypotheses from at least two recognition architectures.
- `PROBABLE`: stable acquired geometry and one strong architecture or two weakly compatible architectures.
- `AMBIGUOUS`: acquired mark exists but segmentation or label remains unstable.
- `UNSEGMENTABLE`: acquired strokes cannot be separated into a defensible character unit.
- `MODEL_ONLY`: prediction lacks adequate source support.
- `NO_SIGNAL`: no acquired stroke support.

## Controls

- spectral-view ablation;
- view-order permutation;
- low-resolution degradation;
- blank-parchment false-glyph control;
- synthetic fading of acquired f116v strokes;
- duplicated-view consistency;
- generic OCR hostile control;
- repeated-glyph morphology consistency.

No absence claim is permitted unless the synthetic-fading control passes at an equal or weaker signal level.

## Decision rules

- No external lexicon, word completion, abbreviation expansion, or language-model correction in extraction.
- Do not insert spaces from linguistic expectation.
- Do not output a polished phrase unless every constituent glyph is individually supported.
- Preserve alternatives as `[a|o]`, `<?>`, `<ligature>`, `<overwritten>`, or `<damaged>`.
- Retracing or multiple-phase claims require differential cross-band behaviour plus stable geometry.

## Deliverables

- `DATA_MANIFEST.csv`
- `MODEL_MANIFEST.json`
- executable pipeline and exact environment
- line crops and support maps
- raw model outputs
- segmentation alternatives
- repeated-glyph clustering results
- `GLYPH_ATLAS.md`
- `TRANSCRIPTION_APPARATUS.md`
- `RESULT.md`
- machine-readable per-glyph JSON
- execution audit with Hugging Face and Fal job IDs

## Compute discipline

Pilot before scaling. Do not launch duplicate jobs. Cancel failed, blocked, or superseded jobs immediately. Ensure no remote job remains running at closeout.