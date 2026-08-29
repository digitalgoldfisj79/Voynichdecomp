# WLCP v0.1 — Word-Length Cryptanalysis Programme

Status: preregistered execution branch, 2026-08-29.

Primary observable: complete running-text token length in ZL3b, analysed under (1) an operational greedy EVA-unit representation and (2) raw transcription-character length as a representation sensitivity bound.

Primary tests: adjacent-token length mutual information; line-initial minus line-final length; whole-folio held-out first-order Markov predictive gain. Each uses 500 deterministic matched permutations within lines. Gate 1 requires |z| >= 2, same direction, and independent replication in both Currier A and Currier B. Gate 2 requires a Gate-1-qualified metric to survive the alternative length representation.

Glyph-length controls test first-unit and final-unit association with total length while permuting length inside Currier × line-position × opposite-edge strata.

Gate 3 is an identifiability gate. A boundary-preserving one-symbol-per-plaintext-symbol substitution and identity/no-cipher necessarily produce exactly the same whole-token length sequence for every plaintext. Consequently no statistic based only on whole-token lengths can distinguish those mechanisms. If Gate 1 passes but Gate 3 fails, the endpoint is WL-1 (structural but non-discriminating). If Gate 1 fails, endpoint is WL-0.

No mechanism exclusion (WL-2) or plaintext-length recovery claim (WL-4) is promoted without an independently fixed plaintext prior/mechanism.

Audit order: circularity → leakage → confounds → matched nulls → control fairness → measurement degeneracy → representation dependence → decision-rule fragility → audit completeness → interpretation.

Atomic checkpoints are pickled after corpus audit, each representation analysis, and gate evaluation. `RESULTS.md` begins with a persistent retracted-findings section.
