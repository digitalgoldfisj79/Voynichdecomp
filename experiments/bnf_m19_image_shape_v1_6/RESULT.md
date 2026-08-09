# BnF M19 Image Shape v1.6 — Result

Date: 2026-08-09
Protocol freeze: `c63624adf065cc361909431c66c932426b1ce96e`
Prospective scaling amendment: `d41b9cd1dce3b933d5b6afa73ed2d36bd1791f7b`
Runner commit: `744f5a81d6db968cdaf60821aaee0e08c28c6999`
HF job: `6a784ba63e1f34a7e32c0200`

Verdict: **SHAPE FEATURES DO NOT STABILIZE M19 SURFACE STATES**.

No full-corpus crop regeneration was authorized. No M19 positive-control language fitting and no Voynich language scoring were run under v1.6.

## Binding S0 feasibility screen

The retained private crop shard contains 12,990 `ccmerge/norm` units from eight pharmaceutical folios. The frozen protocol selected at most 750 units per folio, yielding 6,000 image crops; four folios were ShapeTrain and four ShapeTest by deterministic hash split.

Six EVA-free image feature representations were tested at K={19,25,31,38}: binary topology/geometry (T), HOG (H), normalized raster PCA (R), HOG+topology (HT), raster+topology (RT), and HOG+raster+topology (HRT). Two independent MiniBatchKMeans fits (seeds 408/409) were centroid-matched by Hungarian cosine assignment and evaluated only on ShapeTest.

The prospective gate required assignment stability >=0.75, silhouette >=0.08, >=20 ShapeTest members per cluster, and recurrence in >=2 ShapeTest folios. **0/24 candidates passed.**

Best candidate:

- representation: topology/geometry only (T)
- K=19
- cross-seed ShapeTest assignment stability: **0.61333**
- cosine silhouette: **0.22117**
- minimum cluster size: 68
- minimum folio recurrence: 4/4

The high topology silhouette combined with low assignment stability is informative: there is substantial low-dimensional graphical structure, but it does not resolve into a unique/reproducible 19-state partition under independent fits.

Selected additional candidates:

| representation | K | stability | silhouette | pass |
|---|---:|---:|---:|---|
| T | 19 | **0.6133** | **0.2212** | no |
| R | 31 | 0.5393 | 0.0891 | no |
| HT | 25 | 0.5373 | 0.0748 | no |
| RT | 25 | 0.5220 | 0.1096 | no |
| R | 19 | 0.5180 | 0.0942 | no |
| HRT | 38 | 0.4757 | 0.0628 | no |
| H | 19 | 0.4587 | 0.0810 | no |

## Programme implication

v1.2–v1.4 already showed that DINO CLS/dense and segmental representations recover recurrent boundaries but not stable 19-class identities; v1.5 showed that unconstrained continuous image emissions are not identifiable even on known synthetic M19 controls. v1.6 now shows that conventional binary-shape, HOG, raster and topology constraints do not rescue a global 19/25/31/38-state surface alphabet even on the retained homogeneous eight-folio crop shard.

The next admissible image use is therefore **not another global clustering attempt**. If pursued, image evidence should enter as high-confidence pairwise/local recurrence constraints (must-link/cannot-link or graph regularization) while leaving most instances uncommitted, with a separately qualified synthetic-control instrument. Full-corpus PNG regeneration is not justified merely to rerun hard clustering.
