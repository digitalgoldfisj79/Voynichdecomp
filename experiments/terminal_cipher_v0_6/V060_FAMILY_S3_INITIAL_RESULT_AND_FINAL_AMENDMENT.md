# v0.6 Family S3 — initial joint result and final amendment

Date: 2026-07-16

Verdict: **INITIAL JOINT SOLVER FAILS. SINGLE FINAL AMENDMENT ACTIVATED.**

No test data or Voynich text was scored.

## Initial job

Hugging Face job: `Digitalgoldfish79/6a58920eb1669a49bf077096`

Scientific SHA-256: `cf24de8ef8a52a21abf7527b1d1b0ab86bdcbe1df70b41b04e374ad107a7158a`

## Initial architecture

- ciphertext-only SentencePiece unigram and BPE segmentations;
- vocabulary sizes 48, 63 and 78;
- maximum code length three;
- each candidate mapped to the frozen 63-unit plaintext inventory with the S2 unit-language objective;
- `700,000 × 50` mapping search per candidate;
- fixed code-length, token-count and vocabulary MDL penalties;
- selected segmentation refined at `700,000 × 200`.

## Results

- mean boundary F1: **86.0125%**;
- median boundary F1: **86.9510%**;
- minimum boundary F1: **64.0264%**;
- mean plaintext recovery: **45.7907%**;
- median plaintext recovery: **37.7858%**;
- minimum plaintext recovery: **15.4229%**;
- trials at least 75%: **4/16**;
- exact plaintexts: 0/16.

The boundary statistic narrowly exceeds the aggregate S3 target, but independent tokenisations with similar surface likelihood induce incompatible code dictionaries. Increasing the same mapping search cannot repair a wrong segmentation.

## Final amendment

The sole S3 amendment replaces independent surface segmentation with a joint neural transducer trained only on synthetic train-split examples from the frozen Family S generator.

### Input representation

- unsegmented visible-symbol stream;
- first-occurrence canonicalisation removes arbitrary ten-symbol surface labels;
- observed line boundaries and absolute positions are supplied;
- maximum code length remains three.

### Model

Two independent models with identical architecture and different frozen seeds:

- Transformer encoder-decoder;
- model dimension 384;
- six encoder and six decoder layers;
- eight attention heads;
- feed-forward dimension 1,536;
- character-level plaintext decoder;
- auxiliary boundary classifier on every input position;
- joint loss: plaintext cross-entropy plus 0.3 × boundary binary cross-entropy;
- mixed-precision A100 training.

Each model receives 30,000 updates with effective batch size 32: 960,000 independently generated fresh-codebook examples. Source plaintexts come only from the frozen train split. No development or test sequence is used for training or early stopping. The final checkpoint is selected by update count, not development recovery.

### Inference and lattice refinement

1. ensemble the two models' plaintext and boundary posteriors;
2. decode plaintext with beam width four;
3. derive the eight highest-scoring length-1-to-3 segmentations from the boundary posterior under the frozen code-length prior;
4. solve each segmentation with the unchanged S2 unit-language mapping search at `700,000 × 50`;
5. fully refine the highest train-only-scoring lattice candidate at `700,000 × 200`;
6. select between the direct neural and lattice-refined hypotheses using calibrated model likelihood and unit-language likelihood learned solely from synthetic train examples.

### Gates

The original S3 gates remain unchanged:

- mean plaintext recovery at least 75%;
- median at least 85%;
- at least 13/16 trials at or above 75%;
- mean boundary F1 at least 85%;
- no trial below 40% recovery.

No further amendment is permitted. Failure closes Family S without a locked test. A passing configuration is frozen and evaluated once on untouched test data.