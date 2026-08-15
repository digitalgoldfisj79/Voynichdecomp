# U6 / VTPS Stage-B registration sensitivity v0.3 — closeout

Date: 2026-08-15

## Formal result

**`FAIL_CALIBRATION__ABSTAIN_UNRESOLVED`**

The true retained-vs-switched VTPS target remains sealed:

- `target_opened=false`
- `true_retention_read=false`
- `target_may_open_later=false`

This is a secondary sensitivity experiment. It does not overwrite the preregistered v0.2 primary reconstruction result, which remains **`FAIL_RECONSTRUCTION__ABSTAIN_UNRESOLVED`** under the inherited inlier-ratio threshold of 0.55.

## What the sensitivity experiment resolved

The earlier reconstruction failure was entirely caused by the inherited SIFT inlier-ratio admission gate, not by a failure of the selected homographies on the preserved geometric criteria.

| Admission rule | Folios admitted | Failures |
|---|---:|---|
| inherited 0.55 | 104/107 | f32r, f39r, f40r |
| ratio >= 0.50 | 106/107 | f39r |
| ratio >= 0.45 | 106/107 | f39r |
| ratio >= 0.40 | 107/107 | none |
| preregistered ratio-free geometry | **107/107** | **none** |

The principal ratio-free rule required the exact Yale folio canvas, >=50 absolute inliers, median reprojection <=3 px, and the frozen pipeline's own homography-plausibility predicate. It introduced no replacement ratio threshold.

The exact frozen population was then reconstructed successfully:

- 107/107 folios;
- 9,620/9,620 normalized word crops;
- exact historical crop IDs reproduced;
- word-keyset SHA256 `c494eb695691e899d6e1dc648f9f7d7ec4afe49141a8890f9c1c40638b6a3f84`;
- registration-realization SHA256 `5ef8cdee4defbf0c337a2c718069465e3e0695c1a13b51b248bf983cef901d44`.

Therefore the 0.55 threshold was indeed the sole blocker to entering Stage B.

## Frozen Stage-B result

Stage B ran unchanged on the exact 107-folio population using:

- encoder SHA256 `54ef0612e623fa1755a488cdb975263c93f77c034085b3fa11eff21b62ba52b0`;
- pair-skeleton SHA256 `7f29bb7fe782130ddffe3d7809ce024e04a7eb01fa5c4194440d3be18cea3ed4`;
- 485 calibration pairs;
- 331 valid midfix events;
- 327 valid suffix events;
- 60 development repetitions and 100 frozen confirmation repetitions.

### Midfix

Nuisance control passed. Confirmation FPRs were 0–4%:

- iid 1%;
- page-only 0%;
- hand-only 4%;
- abstract-text-only 1%;
- background-page-PC 1%.

But physical-source power at the frozen beta=0.50 effect size was:

- immediate visual: **45%**;
- line-reset visual: **39%**;
- broad visual: **40%**.

All are well below the preregistered 80% power requirement. Even at beta=0.70 the powers were 69%, 58%, and 69%, respectively.

### Suffix

Nuisance control also passed. Confirmation FPRs were 1–5%:

- iid 1%;
- page-only 3%;
- hand-only 5%;
- abstract-text-only 3%;
- background-page-PC 4%.

Physical-source power at beta=0.50 was:

- immediate visual: **66%**;
- line-reset visual: **49%**;
- broad visual: **58%**.

Again, all are below 80%. At beta=0.70 they rise to 79%, 77%, and 74% respectively, but that stronger effect was not the frozen qualification criterion.

## Scientific interpretation

This is no longer a reconstruction-threshold edge case. The reconstruction problem has been removed cleanly and the exact intended Stage-B experiment has now been executed.

The U6 visual instrument is well controlled against the prescribed nuisance sources, but with 331/327 usable events its current raw 128-D embedding-cosine scalar does **not** provide the prespecified detection power for the synthetic physical-state effects. The failure is substantial for midfix and material for suffix; it is not a 0.79-versus-0.80 technicality at the qualification effect size.

Consequently:

1. no retained-vs-switched target labels may be opened;
2. this result is not evidence for or against a physical production-state counterpart in the Voynich manuscript;
3. the legitimate next scientific route, if pursued, is a separately preregistered **instrument-improvement** experiment while the target remains sealed, not another threshold relaxation.

A defensible next instrument experiment would replace raw embedding cosine with externally calibrated pairwise/style-state scores derived without Voynich target labels, and compare candidate scores solely by nuisance FPR and full power curves on the sealed synthetic Stage-B harness. Any selected scalar would then be frozen before another qualification run.

## Provenance

- v0.3 protocol commit: `91ed8168d4c76b0e4335df4fd6919a5250504b5f`
- v0.3 runner commit: `8d3dd18948ee72c0adff2ab57b804ebf00189744`
- durable result JSON commit: `1887ceebfb7e347e4e08eee3d56f40c82a8e079c`
