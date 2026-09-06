# f116v multispectral recovery and raking programme v0.1

This experiment tests two separate claims against the 2014 f116v image set:

1. whether source-supported enhancement improves the surviving marginal text;
2. whether the damaged field contains reproducible spectral or topographic traces compatible with erased, washed, obscured, or impressed writing.

The scientific verdict is produced only by non-generative operations. Learned super-resolution or diffusion restoration may later be used as labelled visualization, but never to create a stroke or candidate region.

## Branch and protocol

The frozen protocol is `PROTOCOL.md`. Thresholds are derived from known surviving-stroke holdouts, clean parchment controls, band shuffles, leave-family-out runs, and synthetic attenuation controls.

## CPU preflight

```bash
python -m pip install -r requirements.txt
python run_pipeline.py \
  --drive-url 'https://drive.google.com/drive/folders/1mNQGKQDSCR4M_c2M2JrsU5soghvYwMig?usp=sharing' \
  --output results/f116v_real \
  --max-dim 1800 \
  --bootstrap 20
```

The preflight inventories and hashes every source image, selects a visible-light reference, registers bands, creates a normalized spectral cube, runs PCA/ICA/NMF, low-rank residual and RX anomaly analyses, learns an ink-similarity model from visible strokes, and performs bootstrap and leave-family-out stability tests. It also searches metadata for matched opposite-direction raking-light captures.

Source images are never committed or uploaded as workflow artifacts.

## Synthetic falsification controls

```bash
python generate_synthetic_control.py --mode positive --output controls/positive
python generate_synthetic_control.py --mode blank --output controls/blank
python run_pipeline.py --input controls/positive --output results/positive --max-dim 520 --bootstrap 4
python run_pipeline.py --input controls/blank --output results/blank --max-dim 520 --bootstrap 4
```

The positive control contains a faint planted lower-page trace whose spectral response varies by band. The blank control contains the same page texture, stain, noise, visible text, and geometric shifts but no lower-page trace. The registered implementation must nominate candidates in the positive control and none in the blank control.

## Outputs

The central outputs are:

- `inventory.csv` / `inventory.json`;
- `registration.csv` / `registration.json`;
- registered-band, PCA, ICA, NMF, anomaly, restoration, and support panels;
- `candidate_mask.png`, `candidate_overlay.png`, and `candidate_regions.json`;
- `metrics.json` with all thresholds and gate counts;
- `RESULT.md` with separate visible-text, erased-text, and physical-imprint verdicts.

## GPU refinement after the preflight

A positive preflight unlocks a full-resolution, tiled refinement stage. The registered model stack is:

- RoMa v2 for dense cross-band correspondence, with SIFT/ECC retained as an independent registration control;
- DINOv3 dense features for source-supported stroke/background separability;
- DocRes and HAT/DocWaveDiff-class restoration only for side-by-side visualization;
- no OCR, masked-language reconstruction, semantic inpainting, or unlabelled generative completion.

Any model-produced edge must coincide with the non-generative support map. Outputs that do not meet that condition are labelled `VISUALIZATION_ONLY` and excluded from the verdict.
