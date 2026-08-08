# Amendment 001 — computational implementation only

Frozen 2026-08-08 before any language-scoring result was observed.

The first execution (`HF job 6a77a29ada2af92a634efb7f`) was cancelled during language-model construction. Its `build_lm` implementation called `np.bincount(..., minlength=K**4)` once per sentence, causing repeated allocation of a ~331k-bin array. Logs had reached corpus acquisition only; no `DONE` model marker and no language/cipher score had been emitted.

This amendment changes only the implementation of 4-gram count accumulation: collect per-sentence 4-gram indices and call `np.bincount` once per corpus. Corpus choices, folio, transcribers, BnF capacities, cipher models, optimizer parameters, seeds, null procedure and interpretation thresholds are unchanged.
