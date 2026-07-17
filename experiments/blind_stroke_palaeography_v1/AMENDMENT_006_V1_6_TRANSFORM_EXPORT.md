# Amendment 006 — v1.6 transform and provenance export

Date frozen: 2026-07-17

Status: prospective; frozen before any external-corpus image inference.

## Reason for amendment

The completed v1.5.1 run persisted a model checkpoint hash and final page matrices, but source audit after the terminal result established that it did **not** serialize:

- the fitted 512-dimensional whitened PCA;
- the fold-local acquisition residual model;
- the fold-local ink residual model;
- the selected combined nuisance residual model;
- physical page IDs or source-image identifiers alongside the feature rows.

The checkpoint therefore cannot, by itself, reproduce the selected `resid_combined` representation on a new corpus. This is an export/provenance defect, not a change to the frozen scientific target or result.

No HisFrag20, BullingerDB, Voynich, Davis-label or f115r image has been passed through the model at the time of freezing this amendment. Only archive filenames and public metadata have been inspected for corpus suitability.

## Required sequence

1. Complete the exact v1.5.1 reproduction on the same A100 class with the pinned environment.
2. Persist and round-trip verify the selected checkpoint, exact matrices, split manifest, source and logs.
3. Accept the reproduction only under the tolerances frozen in Amendment 005.
4. Run a deterministic **export-only** pass from the accepted checkpoint. The export pass must not perform MAE pretraining, metric learning, checkpoint selection, representation selection or any external-corpus inference.
5. Reconstruct the Historical-WI pages and writer split from the immutable source archive and split manifest.
6. Load the accepted checkpoint, compute the raw page descriptors, refit the already-frozen PCA and residual transforms on the same training writers, and serialize every parameter required for future inference.

## Required exported state

The export bundle must contain at least:

- checkpoint bytes and SHA-256;
- immutable source and helper hashes;
- complete package-version manifest;
- writer split manifest;
- model architecture/configuration manifest;
- NetRVLAD state through the checkpoint;
- PCA `mean_`, `components_`, `explained_variance_`, `explained_variance_ratio_`, `singular_values_`, `n_samples_seen_`, whitening flag and dimensions;
- for each residual model: nuisance mean/std, embedding mean/std, ridge coefficient matrix and alpha;
- frozen selected representation name, `resid_combined`;
- acquisition, ink and combined nuisance definitions and dimensions;
- train, validation and terminal page IDs, writer IDs, image dimensions and deterministic source identifiers aligned row-for-row with exported matrices;
- raw pre-PCA page descriptors and post-transform matrices, or sufficient deterministic provenance to regenerate them exactly;
- file sizes and SHA-256 hashes for every artifact.

## Export acceptance checks

The export is accepted only if:

1. the checkpoint SHA-256 matches the accepted reproduction checkpoint;
2. the writer split bytes are identical to the accepted reproduction split;
3. all exported arrays are finite and have the expected dimensions;
4. applying the exported PCA and residual coefficients reproduces the export-run validation and terminal matrices within maximum absolute error `1e-5`;
5. validation and terminal retrieval metrics for every representation differ from the accepted reproduction by at most `0.001` mAP;
6. the selected representation remains `resid_combined`;
7. all frozen gate decisions remain unchanged;
8. row counts, writer counts and page IDs agree with the source manifest;
9. the complete export bundle is persisted in write-once storage and verified by download and SHA-256 round trip.

Failure of any check blocks external-corpus inference. It does not authorize retuning or a new representation choice.

## External validation remains unchanged

After export acceptance, v1.6 proceeds under Amendment 005. The export pass does not count as independent validation and cannot authorize opening Voynich.

The Voynich seal remains in force:

- no Davis hand labels;
- no Davis five-hand assignments;
- no f115r boundary information;
- no Voynich section or Currier labels;
- no forced K=5;
- no Voynich folio identity used for selection.
