# Pure frozen-DINOv3 CATMuS CTC preflight

## Status

`DINOV3_CATMUS_PREFLIGHT_FAIL`

Final scientific job: `6a71ea466b79c09949c22052`

The preceding launch `6a71e9196b79c09949c22046` failed before model loading because the original SHA bucket rule encountered no eligible test shelfmark in the first 30,000 streamed rows. The corrected sampler used deterministic first-encounter manuscript assignment and completed with disjoint shelfmarks.

## Data

- CATMuS revision: `e11965909ba89dea89476f665fc4d8541b0bf7a1`
- Train/dev/test lines: 256/64/64
- Distinct shelfmarks: 5/2/1
- Centuries: 14–16
- Maximum target length: 48 code points

## Model

- Frozen encoder: `facebook/dinov3-vits16-pretrain-lvd1689m`
- Revision: `114c1379950215c8b35dfcd4e90a5c251dde0d32`
- 64 horizontal patch steps
- Vertical mean/max pooling
- 2-layer bidirectional GRU
- Grapheme CTC decoder, 85 symbols including blank
- No dictionary or language model

## Metrics

| Metric | Untrained | Trained |
|---|---:|---:|
| Development CER | 0.9828 | 0.9528 |
| Test CER | 0.9690 | 0.9607 |
| Test CER without spaces | 0.9630 | 0.9570 |
| Exact test-line accuracy | 0.0000 | 0.0000 |

Blank-control prediction after training: `e`.

Training loss fell from 3.760 to 2.905 over 15 epochs, but held-out improvement was far below the frozen 0.10 absolute gate.

## Interpretation

The implementation learns weak distributional structure, but frozen DINOv3 pooled features alone are not an adequate CATMuS recognizer at this sample size. The failure is consistent with:

1. loss of thin-stroke detail at 16-pixel patch scale;
2. a very small training set relative to the 84-character grapheme inventory;
3. manuscript-level domain shift;
4. CTC blank/common-character collapse.

No f116v output from this model is scientifically admissible.

## Next architecture

Retain frozen DINOv3 as a contextual/cross-view branch and add a trainable narrow convolutional branch over the source pixels. Compare CNN-only against CNN+DINOv3 on the same manuscript-disjoint corpus. DINOv3 is useful only if the fused model improves held-out CER over the pixel-only control.
