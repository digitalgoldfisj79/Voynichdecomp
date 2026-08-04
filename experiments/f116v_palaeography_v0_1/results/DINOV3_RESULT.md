# f116v DINOv3 visual-consistency extension

## Status

`COMPLETE`

Final Hugging Face job: `6a71e3af6b79c09949c21ffa`

- Hardware: one L4 GPU
- Running time: 48 seconds; 115 seconds including scheduling
- Terminal state: completed
- Positions tested: 29 frozen probable glyph windows
- Source views: laboratory true colour, expert monochrome multispectral PCA, expert colour PCA

The official gated DINOv3 checkpoint was accessed through the account HF token. The earlier DINOv3 failure was therefore an authentication failure, not a scientific failure.

## Models

- DINOv3: `facebook/dinov3-vitb16-pretrain-lvd1689m`
  - revision: `5931719e67bbdb9737e363e781fb0c67687896bc`
- DINOv2: `facebook/dinov2-base`
  - revision: `f9e44c814b77203eaa57a6bdbbd535f21ede1415`

Both models were used only as unlabelled visual encoders. OCR labels did not form embeddings, select nearest neighbours, or determine clusters.

## Primary cross-view retrieval

### True colour to monochrome PCA

| Metric | DINOv3 | DINOv2 | Difference |
|---|---:|---:|---:|
| Exact-position top-1 retrieval | 0.276 | 0.207 | +0.069 |
| Aligned-vs-mismatched AUC | 0.877 | 0.792 | +0.084 |
| Median aligned similarity | 0.769 | 0.780 | -0.011 |
| Median mismatched similarity | 0.668 | 0.680 | -0.011 |
| Median aligned-minus-mismatched margin | 0.1004 | 0.1001 | +0.0002 |

DINOv3 is materially better at separating the correct cross-view position from mismatched glyph windows. Exact top-1 remains only 8/29, so this is useful localisation evidence rather than a solved glyph inventory.

### True colour to colour PCA

| Metric | DINOv3 | DINOv2 |
|---|---:|---:|
| Exact-position top-1 retrieval | 0.414 | 0.069 |
| Aligned-vs-mismatched AUC | 0.824 | 0.669 |
| Median aligned-minus-mismatched margin | 0.071 | 0.066 |

This is the clearest DINOv3 gain. The colour-PCA rendering was weak for OCR but retains spatial visual information that DINOv3 can match substantially better than DINOv2.

## Dense patch correspondence

Mutual nearest-neighbour DINO patch tokens were tested only on the 30% most edge-rich line patches. `Identity fraction` is the fraction of mutual matches within one patch cell of the known alignment. A deterministic quarter-line shift was the negative control.

| Line | Pair | DINOv3 mutual matches | DINOv3 identity | DINOv2 identity | Shift control |
|---|---|---:|---:|---:|---:|
| 1 | true/BW | 133 | 0.962 | 0.500 | 0.000 |
| 2 | true/BW | 174 | 0.994 | 0.777 | 0.000 |
| 3 | true/BW | 145 | 0.959 | 0.719 | 0.000 |
| 4 | true/BW | 185 | 0.989 | 0.720 | 0.000–0.008 |
| 1 | true/colour-PCA | 101 | 0.980 | 0.633 | 0.000 |
| 2 | true/colour-PCA | 135 | 0.970 | 0.693 | 0.000 |
| 3 | true/colour-PCA | 83 | 0.843 | 0.560 | 0.000–0.013 |
| 4 | true/colour-PCA | 110 | 0.955 | 0.595 | 0.000 |

This is strong evidence that the three processed images preserve the same local physical stroke structures. It validates the use of DINOv3 for registration, source-support mapping and crop retrieval.

It does not identify the strokes as particular letters.

## Post-hoc comparison with provisional CATMuS labels

The frozen CATMuS labels were revealed only after DINO retrieval.

| Metric | DINOv3 | DINOv2 |
|---|---:|---:|
| Cross-view nearest-neighbour label agreement | 0.364 | 0.273 |
| Within-view repeated-label top-1 | 0.000 | 0.045 |
| Same-label-vs-different-label AUC | 0.417 | 0.432 |

The repeated-label metrics are at or below chance. Thus DINOv3 does **not** independently support the proposition that the CATMuS-labelled `a`, `o`, `r`, `e`, `u`, or `c` instances form reliable repeated glyph classes.

This is the crucial negative result. DINOv3 validates cross-view stroke localisation much more strongly than it validates OCR character identity.

## Consequences for the palaeographic apparatus

1. The writing and its local stroke positions are real across the acquired views; they are not processing artefacts.
2. DINOv3 materially improves correspondence between true colour and both PCA products.
3. The line-2 region previously rendered approximately as `…chico<n?> [o|e]ladaba…` remains an OCR-derived label hypothesis.
4. DINOv3 does not upgrade that sequence to an independently supported transcription.
5. The correct next use of DINOv3 is source-to-source retrieval and comparator-manuscript matching, not asking it to read the line.

## Evidence discipline

No OCR, dictionary, language model, word completion, abbreviation expansion, diffusion restoration, super-resolution, or semantic inpainting was used in the DINOv3 tests. Provisional CATMuS labels were consulted only after all visual retrieval outputs had been computed.
