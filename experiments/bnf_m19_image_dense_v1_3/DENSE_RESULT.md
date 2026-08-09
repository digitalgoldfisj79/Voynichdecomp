# BnF M19 Image Bridge v1.3 — Dense result

HF job: `6a783d8cda2af92a634f0008`
Runner commit: `80d081c6ac9281e6e7b52e704d8d30342c7d2267`
Protocol freeze: `53e5aa991832738c14b6c25c77f784d882dc6867`

Dense patch-mean DINO improves the image-only partition substantially relative to CLS but does not meet the frozen 0.75 assignment-stability gate. No M19 language score was generated and H12/C12 stayed sealed.

| representation | K | stability | coverage | silhouette | gate |
|---|---:|---:|---:|---:|---|
| raw dense | 19 | **0.6385** | 0.9476 | **0.11931** | fail |
| raw dense | 25 | 0.5540 | 0.9476 | 0.11326 | fail |
| raw dense | 31 | 0.5177 | 0.9457 | 0.09013 | fail |
| raw dense | 38 | 0.4485 | 0.9474 | 0.08706 | fail |
| folio-centred dense | 19 | 0.5487 | 0.9521 | 0.10119 | fail |
| folio-centred dense | 25 | 0.4998 | 0.9515 | 0.09858 | fail |
| folio-centred dense | 31 | 0.5535 | 0.9510 | 0.08030 | fail |
| folio-centred dense | 38 | 0.5024 | 0.9514 | 0.08744 | fail |

Per the frozen v1.3 protocol, the CLS+dense hybrid fallback is now admissible.
