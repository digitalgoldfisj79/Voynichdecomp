# BnF M19 Image Bridge v1.2 — Arm B result

HF job: `6a7839e9da2af92a634effda`
Runner: `run_arm_b_qualified.py` at commit `32858352f5a35fcdbc7c6128a450bd36cbbbb6fe`
Protocol freeze: `d230bf0dcbef563993b257637901a249fbe5ae16`

Verdict: **ARM B IMAGE-UNDERPOWERED**.

No language score was generated; the synthetic M19 qualification, H12 and C12 remained locked because the visual segmental alphabet never passed its prospective stability gate.

Using folio-centred DINO CLS, K=19, adjacent `ccmerge` components could merge in groups of 1–3. All settings produced highly recurrent segments and ~95% Tvis acceptance. Boundaries were substantially more reproducible than Arm-A class identities, but the 19 visual class labels remained unstable across independent fits.

| lambda | boundary F1 | class agreement / stability | Tvis coverage | silhouette | mean segments/word | gate |
|---:|---:|---:|---:|---:|---:|---|
| 0.02 | 0.8296 | 0.6281 | 0.9494 | 0.09386 | 3.390 | fail |
| 0.04 | **0.8431** | **0.6680** | 0.9480 | 0.09678 | 3.379 | fail |
| 0.06 | 0.8312 | 0.6171 | 0.9492 | 0.09547 | 3.360 | fail |
| 0.08 | 0.8386 | 0.5940 | **0.9521** | **0.09932** | 3.345 | fail |
| 0.10 | 0.8379 | 0.6192 | 0.9479 | 0.09927 | 3.325 | fail |
| 0.12 | 0.8337 | 0.5423 | 0.9479 | 0.09236 | 3.317 | fail |

Frozen stability threshold was 0.75. The result supports an image-derived recurrent segmentation structure, but not a sufficiently stable 19-class surface alphabet under CLS features.
