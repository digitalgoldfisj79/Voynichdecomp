# Amendment 005 — Q3 training-half support-complete qualification

Date: 2026-08-09
Status: **prospective with respect to all binding qualification evidence and all Voynich H17/C17 scoring.**

The Q2 qualification run was cancelled before completing K=26 and before any Voynich H17/C17 language score. A control-only diagnostic showed that the selected Arabic K=26 84,000-letter span contained two normalized `o` characters, but both occurred in the held-out 39,000-letter portion. The 45,000-letter fitting portion therefore had zero capacity to emit BnF numerical value 22, making legal 19-value key generation impossible despite whole-span support.

The binding qualification is restarted under a fresh namespace `M19STAv17Q3` with the following single correction:

- candidate control spans are accepted only if the **45,000-letter fitting half itself** contains plaintext-letter support capable of emitting every one of the 19 frozen BnF numerical values;
- the 39,000-letter held-out half remains untouched and is not used to fit the map;
- span candidates remain drawn exclusively from the same untouched UD dev+test pools;
- the first deterministic support-complete candidate under the Q3 SHA-256 namespace is selected.

All K=22, K=26 and K=36 controls are rerun from scratch. Q1/Q2 control outcomes are development-only and do not qualify the instrument.

The vectorized full-score optimizer, proposal kernel, annealing temperature schedule, step counts, restart counts, deterministic polish, source files, language models, BnF channel, representation definitions, 60/20/20 RF split, and all success thresholds are unchanged from the frozen protocol plus Amendments 001–004.

No RF H17/C17 score may be generated unless all three Q3 K-specific qualification gates pass.
