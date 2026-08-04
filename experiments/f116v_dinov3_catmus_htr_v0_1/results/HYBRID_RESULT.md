# Hybrid pixel + DINOv3 CATMuS result

## Status

`HYBRID_DINOV3_CATMUS_PASS`

Final job: `6a71ebd36b79c09949c22062`

- Hardware: one L4 GPU
- Running time: 168 seconds
- CATMuS revision: `e11965909ba89dea89476f665fc4d8541b0bf7a1`
- DINOv3-S/16 revision: `114c1379950215c8b35dfcd4e90a5c251dde0d32`
- Train/dev/test lines: 512/96/96
- Distinct shelfmarks: 13/2/2
- All shelfmark sets disjoint

## Comparison

| Arm | Development CER | Test CER | Test CER without spaces | Blank prediction |
|---|---:|---:|---:|---|
| CNN-only | 0.9771 | 0.9656 | 0.9585 | `e` |
| CNN + frozen DINOv3 | 0.6836 | 0.5952 | 0.6828 | `PH` |

The fused model improved test CER by **0.3704 absolute** over the pixel-only arm and passed the preregistered development, comparative and blank-control gates.

## Interpretation

This is a genuine architecture-level result: frozen DINOv3 features provide transferable information that the narrow pixel branch did not learn from 512 CATMuS lines alone.

It is not a strong general medieval recognizer. A 0.595 test CER means that roughly three of every five held-out characters still require insertion, deletion or substitution. Exact held-out line accuracy remained zero.

The model is therefore suitable for:

- proposing alternative glyph boundaries and low-level character hypotheses;
- testing whether the same local sequence recurs across f116v views;
- prioritising regions for human palaeographic review.

It is not suitable for:

- producing a polished f116v sentence;
- identifying language from its output;
- overriding source pixels or human palaeography;
- serving as independent confirmation of Kraken–CATMuS, because both systems are supervised by CATMuS graphematic labels.

No dictionary, language model, abbreviation expansion or word correction was used.
