# Amendment 007A — non-primary diagnostic split correction

Date: 2026-08-08
Timing: frozen before execution of the Stage 5 confound job and before any VMS similarity.

Amendment 007 described the leakage-prone crop-random diagnostic as stratified 5-fold. Two retained manuscript classes contain only two crops, so five stratified folds are mathematically impossible.

The executable therefore uses:

`n_splits = min(5, minimum manuscript crop count)`

which is 2 for the frozen 59-crop confound subset.

This correction affects only the explicitly non-primary crop-random diagnostic. The primary gate remains leave-one-source-page-out manuscript classification with macro one-vs-rest AUC and is unchanged.
