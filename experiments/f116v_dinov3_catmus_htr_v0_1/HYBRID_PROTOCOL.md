# Hybrid pixel + DINOv3 CATMuS HTR protocol

## Purpose

Test whether DINOv3 contributes useful contextual information when thin pen-stroke detail is preserved by a trainable pixel branch.

## Arms

Both arms use the same CATMuS lines, shelfmark-disjoint partitions, grapheme vocabulary, bidirectional GRU, CTC loss and optimization schedule.

1. `CNN_ONLY`: four stride-2 convolution blocks reduce a 128 × 1024 grayscale line to a 64-step pixel sequence.
2. `CNN_DINOV3`: the same pixel sequence is concatenated with projected frozen DINOv3-S/16 mean/max patch features at the corresponding 64 horizontal positions.

DINOv3 is useful only if the fused arm improves held-out manuscript CER over `CNN_ONLY`.

## Data and compute

- 512 train / 96 development / 96 test lines.
- 14th–16th century CATMuS `DefaultLine` records.
- Maximum 48 Unicode code points; all targets must be CTC-feasible.
- Shelfmarks assigned deterministically and permanently to one partition.
- One L4 GPU; hard timeout 60 minutes.
- Frozen DINOv3 features extracted once.

## Optimization

- Classifier blank bias initialized to -2.
- Standard CTC loss.
- A small temporary blank-probability penalty is applied during early epochs to prevent the degenerate all-blank solution; it decays to zero.
- No lexicon, language model, abbreviation expansion or word correction.

## Pass gate

The hybrid pilot passes only if:

1. all shelfmark sets are disjoint;
2. both arms train with finite loss;
3. fused development CER is below 0.90;
4. fused test CER beats CNN-only test CER by at least 0.02 absolute;
5. the blank control emits at most two non-space characters.

Only a passing fused model may be applied to f116v, and its f116v output remains a CATMuS-supervised model hypothesis rather than independent palaeographic confirmation.
