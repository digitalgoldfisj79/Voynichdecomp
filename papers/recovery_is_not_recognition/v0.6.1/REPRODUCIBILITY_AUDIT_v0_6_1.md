# Reproducibility audit v0.6.1

## Verdict

The v0.6 manuscript understates the surviving evidential record. The phrase “some inherited results are documented through recorded summaries rather than a single end-to-end independent rerun” conflates a deliberately filtered reviewer archive with the complete versioned research repository.

The 18 experiments in Supplementary Table S1 are not one monolithic pipeline and do not require one monolithic rerun. Each must instead have a traceable chain from protocol and input through implementation to result and verification.

Repository-wide inspection produces the following classification:

- **Committed machine or row-level evidence, or an exact/scientifically identical clean rerun:** rows 1, 4, 5, 10, 12, 14, 15, 16, 17 and 18. Row 1 has complete committed inputs, implementations and cached outputs; a partial clean-clone confirmation was cancelled once it had established executability because regenerating all 23 models was non-decisive and computationally redundant.
- **Committed code, deterministic inputs, protocol, aggregate result and provenance, but not every historical row-level output:** rows 6, 7, 8, 9 and 13.
- **Committed aggregate machine summary without all final rows:** row 11.
- **Complete hash-verified research archive outside GitHub, filtered for reviewer distribution:** row 3.
- **Principal genuine source-code gap:** row 2, the exact original ten-feature C2ST construction harness.

The terminal blind-recognition experiment has been cleanly recomputed from its branch. The rerun reproduced the registered scientific hash `63d1ffac893e2e0449a22c87d5ff92aee2ab2c6d02a28f93cad8980204091add`, the complete confusion matrix, macro-AUC `0.9650487076`, P precision `0.9615384615`, P recall `0.3472222222`, structured-control false-positive rate `0`, and ECE `0.06070659285`.

The original H2 archive has also been verified against its external SHA-256 and internal checksum manifest. It contains the analysis code, result objects, derived data and authoritative ledger used for the within-line order panel. Its omission from the reviewer ZIP was a redistribution decision, not loss of the experiment.

## Consequence for the manuscript

The manuscript should remove the generic “inherited summary” sentence. It should instead state precisely that:

1. the experiments are preserved across versioned branches rather than one unified preregistration;
2. the reviewer archive is a filtered submission snapshot rather than the entire repository;
3. most headline results have committed machine outputs, row-level outputs, clean-environment recomputations, or a combination of those;
4. five earlier computationally intensive calibrations retain exact implementations, deterministic inputs, reports, job identifiers and hashes but not committed copies of every historical row;
5. the exact original C2ST feature-construction code is the principal unresolved artifact gap;
6. no external research team has independently replicated the complete programme.

## Recommended final wording

### Limitations

> The study was sequentially frozen rather than governed by one comprehensive preregistration. The early generator hierarchy was exploratory; later adversarial and recovery tests were frozen separately. One shifting-alphabet development benchmark was exposed before the complete development set, although no tuning followed and the untouched test remained sealed. Design, implementation and interpretation occurred within one AI-assisted research workflow. Clean-clone reruns and separately implemented internal audits therefore test computational consistency but do not constitute external independent replication. Conclusions remain bounded to the recorded implementations, representations and controls.

### Data availability and reproducibility

> The experiments are preserved across versioned repository branches containing protocols, executable implementations, input manifests or deterministic input generators, result reports, commit and job provenance, and scientific hashes. Machine-readable or row-level outputs are committed for the generator hierarchy, generator-disjoint latent-order test, terminal transposition and polygraphic tests, compression experiments and CoReMA analysis; the terminal recognition gate and charged source-transfer comparison also have clean-environment recomputation records. Several earlier computationally intensive calibration runs retain their exact code, inputs, aggregate results and immutable job hashes but not committed copies of every historical row-level file. The accompanying reviewer archive is a filtered submission snapshot and does not duplicate the complete Git history. The exact original ten-feature classifier two-sample-test construction script is the principal unresolved artifact gap; its numerical record is retained, but an external exact rerun requires recovery of that script or a transparently labelled reconstruction. Copyrighted research copies and large model artefacts are excluded or represented by immutable identifiers and hashes. External third-party replication has not yet been completed.
