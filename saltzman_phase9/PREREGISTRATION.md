# Voynich Consequence Panel v0.1 — preregistration

The confirmatory metric panel was sealed while Phase 6 was still scoring. This execution protocol was frozen before Phase 8 ordering results were visible.

## Confirmatory question

Does the final frozen ReM → `SWITCH_LINE` → K2 production architecture improve **independent** Voynich structural statistics relative to the cipher alone?

The four discovery statistics `ED1_N0`, `ED1_N1`, `ED1_N3`, and `E1_N0` are excluded from the confirmatory verdict.

## Inputs and baseline

Use the existing repository transcription `Paper/Cipher_paper/enriched_records.pkl` and the existing 85-metric implementation `Paper/Cipher_paper/S5_score_85_metrics.py`. Their Git blobs are frozen in `protocol.json`. Build the Voynich target once, using the real manuscript lineation reconstructed from `(folio,line_no)` and the pre-existing metric tolerances. No target is re-estimated after synthetic scoring.

## Synthetic programme

Use all 190 canonical ReM documents, first 2000 tokens, W=10, three locked production replicates (0,1,2), ATOMIC and LITERAL representations, and the exact Phase-5 plan/state seed namespaces. The canonical fixed-tau=3 reset/order pipeline is selected only by the separately frozen `FINAL_PIPELINE_SELECTION_RULE.json`; Phase-9 scores cannot change that selection.

Each synthetic text is scored with the existing metric function at 100 background subsamples, 200 words per subsample, Levenshtein disabled because none of the 24 locked metrics requires it. Metric RNG is common between cipher-only and final pipeline.

For each of the 24 metrics, normalize absolute error to the existing tolerance and take the worse ATOMIC/LITERAL loss. Per-document arm loss is the median over all 24 metrics. The primary paired gain is cipher-only loss minus final-pipeline loss.

## Primary adjudication

Bootstrap the median document gain over ReM documents with 10,000 resamples. `CONFIRMATORY_BROAD_IMPROVEMENT` requires lower CI > 0, positive median gain in at least five of six locked metric families, and no family median gain below -0.5 tolerance units. `CONFIRMATORY_NO_GAIN` requires upper CI <= 0. All other results are `CONFIRMATORY_NARROW_OR_MIXED`.

If reset/order remained unresolved upstream, the predeclared alternate sensitivity pipeline(s) are scored but can never replace the canonical primary arm on the basis of Phase-9 results.

## Boundary holdout

As a secondary prediction, compare each real or synthetic line boundary `(previous last token, next first token)` to the matched within-line endpoint pair `(previous penultimate token, previous last token)` for exact repetition and absolute word-length difference. Real Voynich boundaries are restricted to consecutive lines in the same folio. These diagnostics are reported only; they do not select a model.
