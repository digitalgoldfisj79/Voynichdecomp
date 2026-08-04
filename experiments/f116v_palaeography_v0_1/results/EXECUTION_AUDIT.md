# f116v palaeography execution audit

Date: 2026-08-04

## Final branch

- Repository: `digitalgoldfisj79/Voynichdecomp`
- Branch: `experiment/f116v-palaeography-v0.1-20260804`
- Final corrected execution code:
  - `extract_glyphs.py`
  - `extract_glyphs_v2.py`
- Gate-correction commit: `df48335f183b3e989043fc8c0ff655c06a7382c4`

## Source assets

| Key | Source | Drive ID | SHA-256 |
|---|---|---|---|
| true | `Lab_true_color_TIFF/Voynich_116v_PSC.tif` | `1EwdxnZURhNOjLwCTiaVZVMPW0UDeNPIK` | `45ff6bb69a24ab8ec1826fa99dd2793d9a037f400cb5db059624fbe5bde2a8e1` |
| bw | expert rotated monochrome PCA TIFF | `16SuJ5R7RpPKXRnySPv8Pn1tNE0WouTGF` | `c82b8e7e7db10fc92b97dc7a4b5dfa46b83312236f8ffec0bab63599bb14371d` |
| pca | expert colour PCA TIFF | `1Ed7oVeeOSEawpizLi8eOu47ZFR6WYQsg` | `c1edc076cdf77bb483029fc00513a93023f2444820355582e73d89b51d0b0988` |

All three decoded to 8176 × 6132 RGB source arrays before common-frame reduction.

## Remote jobs

### Discovery and implementation checks

- `6a71ceaea00abefd4b2915b4`: CATMuS Zenodo/Kraken discovery.
- `6a71cfa86b79c09949c21baa`: image decoding/localisation attempt; failed because the minimal TIFF stack lacked LZW codec.
- `6a71cffca00abefd4b2915d7`: successful TIFF decoding and preliminary line localisation.
- `6a71d4c26b79c09949c21d2c`: Kraken direct-recognition API inspection.
- `6a71d77da00abefd4b29199f`: rectangular line-box recognizer test; completed with a model-interface warning.
- `6a71d80b6b79c09949c21ef8`: corrected baseline-line test.

### Whole-line hostile pilot

- `6a71d2cba00abefd4b29166f`
- Hardware: L4 GPU
- Status: completed
- Result: `PILOT_STABILITY_GATE_FAIL`
- Reason: unconstrained page-style segmentation generated large pseudo-text outputs on blank parchment and unstable outputs across views.

### Baseline-constrained extraction v0.1

- `6a71da0aa00abefd4b2919c9`
- Hardware: L4 GPU
- Status: completed
- Scientific output: useful positional agreements, but the interval-versus-ring physical gate was shown to be invalid for connected cursive because adjacent strokes contaminate the ring.
- DINOv3 attempt failed because the official repository was gated.

### Corrected final extraction

- `6a71db2e6b79c09949c21f1c`
- Hardware: L4 GPU
- Status: completed
- Running time: 99 seconds; 156 seconds including scheduling.
- Result: `GLYPH_EXTRACTION_PILOT_PASS`
- Visual encoder: ungated DINOv2 base.

## Corrections made openly

1. The first whole-line pipeline allowed Kraken to segment a noisy crop as a page. Its output was rejected rather than interpreted.
2. CATMuS expects a baseline polygon. The direct recognizer was rerun with an explicit `BaselineLine` and full character cuts/confidences.
3. The initial per-character ink-effect metric compared a connected glyph with a ring containing its neighbours. It was replaced by blank-calibrated dual-view confidence plus local edge correlation.
4. Gated DINOv3 was replaced by ungated DINOv2 for visual comparison only.

## Terminal compute state

At closeout, Hugging Face Jobs reported **no running jobs**.
