# DINOv3–CATMuS HTR pilot v0.1

## Objective

Test whether a frozen DINOv3 visual encoder can be converted into a useful medieval handwriting recognizer by training a small 2D-to-1D temporal CTC head on CATMuS Medieval line images.

This is a feasibility experiment. It does not authorize a complete f116v transcription.

## Frozen architecture

1. Encoder: `facebook/dinov3-vits16-pretrain-lvd1689m`, frozen.
2. Input: grayscale manuscript line rendered as RGB, 128 px high, at most 1024 px wide, right-padded with parchment-white pixels.
3. Patch sequence: take the final DINOv3 patch tokens, reshape to the 8 × 64 patch grid, concatenate vertical mean and maximum features, producing a 64-step horizontal sequence.
4. Projection: LayerNorm → linear 2D-to-256 → GELU → dropout.
5. Sequence model: two-layer bidirectional GRU, 192 hidden units per direction.
6. Decoder: linear layer into a character inventory plus CTC blank.
7. No dictionary, external language model, abbreviation expansion or word-level correction.

## Data

- Dataset: `CATMuS/medieval`, CC BY 4.0.
- Eligible lines: 14th–16th century, `DefaultLine`, 8–48 Unicode code points, and CTC-feasible at 64 time steps.
- The split is recomputed from a SHA-256 hash of `shelfmark`; a shelfmark may occur in exactly one of train, development or test.
- Images and text remain paired; no OCR output is used as training truth.

## Staged execution

### Preflight

- 256 train / 64 development / 64 test lines.
- Up to 15 epochs with early stopping.
- One L4 GPU, hard timeout 45 minutes.

The preflight passes only if:

1. all three shelfmark sets are disjoint;
2. CTC loss remains finite;
3. trained test CER improves by at least 0.10 absolute over the untrained head;
4. trained development CER is below 0.95;
5. no image/text length incompatibility is silently discarded after the manifest is frozen.

### Scale-up

Only after the preflight passes:

- 1,500 train / 200 development / 200 test lines;
- compare against the existing Kraken–CATMuS recognizer on the same held-out manuscripts;
- run the trained DINOv3–CTC head on the fixed f116v line crops;
- treat f116v outputs as model hypotheses unless independently corroborated.

## Controls and reporting

- Report untrained-head CER, trained CER, exact-line accuracy and space-stripped CER.
- Preserve per-line predictions and shelfmarks.
- Record DINOv3 revision, dataset revision, package versions, hardware and job IDs.
- A trained DINOv3 head is not independent of CATMuS supervision; it is an architecture-level replication using the same graphematic standard.
- Do not promote an f116v glyph to `SUPPORTED` solely because Kraken–CATMuS and DINOv3–CATMuS agree: both ultimately learn from CATMuS labels.
