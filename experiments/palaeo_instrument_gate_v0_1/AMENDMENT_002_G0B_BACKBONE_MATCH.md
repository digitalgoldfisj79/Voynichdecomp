# Amendment 002 — G0B backbone-match correction

**Date:** 2026-08-08

**Status:** frozen after G0A and amended G0C were observed, but before any DINOv2 G0B metric was computed.

## Trigger

The v0.1 protocol treated the sealed Stage-5 `inkmask_v1` confound failure as binding G0B evidence. Stage 5 used `facebook/dinov3-vit7b16-pretrain-lvd1689m`, while G0A and G0C in this programme use `facebook/dinov2-small` at immutable revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`.

A nuisance failure cannot be transferred automatically across different backbones. Therefore the original G0B declaration is corrected prospectively.

## Corrected G0B

Rerun the exact sealed Stage-5 confound design on the same 59-row manifest with DINOv2-small:

- same 59 crop coordinates and source pages;
- same three image variants: `rgb_norm_v1`, `gray_bgdiv_v1`, `inkmask_v1`;
- same page-held-out manuscript-identity prediction;
- same train-only standardisation and balanced multinomial logistic regression;
- same macro one-vs-rest AUC and top-1 metrics;
- same frozen thresholds: PASS <=0.65; CAUTION >0.65 and <=0.70; FAIL >0.70.

The primary G0B decision is the `inkmask_v1` decision. The RGB and greyscale variants remain diagnostics.

No VMS/corridor target similarity is permitted.
