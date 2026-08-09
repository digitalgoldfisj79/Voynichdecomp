# BnF M19 Image Bridge v1.3 — Hybrid result

HF job: `6a783eda3e1f34a7e32c0118`
Runner commit: `8ddef57ff407f6eb641d1a1afcb4214b645b7036`
Protocol freeze: `53e5aa991832738c14b6c25c77f784d882dc6867`

Verdict: **DENSE/HYBRID IMAGE-UNDERPOWERED**.

The frozen CLS+dense hybrid fallback did not meet the 0.75 visual assignment-stability gate. No M19 language score was generated; H12 and C12 remained sealed.

| hybrid representation | K | stability | coverage | silhouette | gate |
|---|---:|---:|---:|---:|---|
| raw | 19 | 0.5229 | 0.9465 | 0.11567 | fail |
| raw | 25 | 0.4141 | 0.9481 | 0.09406 | fail |
| raw | 31 | 0.4754 | 0.9492 | 0.08219 | fail |
| raw | 38 | 0.4639 | 0.9460 | 0.09519 | fail |
| folio-centred | 19 | 0.5619 | 0.9508 | 0.10802 | fail |
| folio-centred | 25 | **0.5814** | 0.9519 | 0.10202 | fail |
| folio-centred | 31 | 0.4676 | 0.9509 | 0.09167 | fail |
| folio-centred | 38 | 0.5095 | 0.9498 | 0.08849 | fail |

Across v1.2/v1.3, dense raw K=19 remains the strongest hard image-class substrate (stability 0.6385, silhouette 0.11931), while CLS segmental merging produced reproducible boundaries but unstable class identities. A subsequent experiment may therefore combine dense embeddings with visual-only segmental merging, but that is a new preregistered model rather than a relaxation of this result.
