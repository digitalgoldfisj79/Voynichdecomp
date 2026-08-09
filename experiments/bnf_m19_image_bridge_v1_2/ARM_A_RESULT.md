# BnF M19 Image Bridge v1.2 — Arm A result

HF job: `6a783867da2af92a634effcf`
Runner commit: `d4675efc01928ffac79ea605dc223628092cbbda`
Protocol freeze: `d230bf0dcbef563993b257637901a249fbe5ae16`

Verdict: **ARM A IMAGE-UNDERPOWERED**.

The runner loaded 332,085 `ccmerge/norm` image components across 225 folios from the frozen private DINOv3 archive. The deterministic image-only split was 112 T12 / 45 H12 / 68 C12, with T12 further split 90 Tfit / 22 Tvis.

No language score was generated. H12 and C12 remained sealed.

Visual candidates:

| representation | K | cross-seed stability | Tvis acceptance | cosine silhouette | gate |
|---|---:|---:|---:|---:|---|
| raw CLS | 19 | 0.4453 | 0.9462 | 0.10048 | fail |
| raw CLS | 25 | 0.5724 | 0.9475 | 0.09343 | fail |
| raw CLS | 31 | 0.4647 | 0.9463 | 0.07870 | fail |
| raw CLS | 38 | 0.5079 | 0.9475 | 0.07547 | fail |
| folio-centred CLS | 19 | 0.5497 | 0.9512 | **0.10710** | fail |
| folio-centred CLS | 25 | **0.5948** | 0.9500 | 0.10602 | fail |
| folio-centred CLS | 31 | 0.4662 | 0.9504 | 0.09413 | fail |
| folio-centred CLS | 38 | 0.4208 | 0.9494 | 0.07533 | fail |

All candidates had broad cross-folio recurrence and high Tvis coverage, but none met the prospectively frozen stability requirement of 0.75. By the frozen selection rule, the fallback substrate for Arm B is folio-centred K=19 because it has the highest Tvis silhouette; Arm B must independently pass its own visual gate after segmental merging.
