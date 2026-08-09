# BnF M19 Image Bridge v1.3 — Dense/Hybrid DINO Protocol

Date: 2026-08-09
Parent: v1.2 Arm A/B bounded negative.
HF source revision: `Digitalgoldfish79/vdino3-crops@ea597db8ff2c06631c4c311d90c8cf0418f5e26c`.

## Motivation

v1.2 showed that CLS embeddings support recurrent image segmentation but do not yield a stable 19–38-class visual alphabet. The same frozen corpus already contains full-corpus spatial patch-mean DINO embeddings (`corpus_embeddings_full_dense.npz`). These are a prospectively distinct visual representation; no v1.2 Voynich language score was ever generated.

## Sealing and split

All v1.2 sealing rules remain binding. In particular `word`, `eva_aligned`, `eva_glyph`, `word_len` and transliteration strings are forbidden until terminal post-verdict audit.

Use the exact v1.2 SHA-256 split: 112 T12 / 45 H12 / 68 C12, with the same 90/22 Tfit/Tvis subdivision. C12 remains language-sealed until an H12 primary gate passes.

Primary image stream remains `ccmerge`, `view=norm`, `low_conf=false`.

## Visual ladder

### D — dense patch-mean

Use the 768-dimensional spatial patch-mean vectors from `results/corpus_embeddings_full_dense.npz`.

Test:
- D0: unit-normalized dense vector;
- D1: folio-centred dense residual.

For K in {19,25,31,38}, run the exact v1.2 Arm-A visual gate: two independent MiniBatchKMeans fits, Hungarian centroid matching, Tvis assignment stability, cosine silhouette, 5th-percentile acceptance, cluster recurrence/count floors. Pass thresholds remain stability >=0.75, accepted Tvis fraction >=0.75, every cluster in >=3 Tvis folios and >=25 accepted assignments.

Choose maximum Tvis silhouette among passing D candidates; ties within 0.005 choose smaller K then D1. If at least one D candidate passes, do not inspect hybrid features.

### H — CLS+dense hybrid fallback

Only if no D candidate passes, load the matching CLS vectors and form `normalize(CLS + dense)` for each image component, then test:
- H0: raw hybrid;
- H1: folio-centred hybrid.

Use the same K panel, visual gates and selection rule. If no H candidate passes, verdict is `DENSE/HYBRID IMAGE-UNDERPOWERED`; no language score is admissible.

## M19 solver and gates

If a visual candidate passes, use the exact v1.2 Arm-A M19 solver and six-language positive-control qualification. The selected K determines the legal surjective K->19 numerical map (one or two surface classes per value).

Qualification and H12/C12 thresholds are unchanged from v1.2:
- positive controls: 6/6 language correct; min margin >=0.05; median map recovery >=0.95; min recovery >=0.85; min independent-fit agreement >=0.90;
- H12: rank 1, margin >=0.05 nats/mapped unit, fit agreement >=0.90, coverage >=0.90;
- C12, fixed map/no refit: rank 1, margin >=0.05, coverage >=0.90, positive candidate-vs-best-other margin in all four frozen C12 buckets.

No decoded strings may be inspected before C12 confirmation.

## If confirmed

Only after C12 confirmation, run the v1.2 order/frequency nulls and cross-representation replication. EVA fields remain sealed until those tests finish.
