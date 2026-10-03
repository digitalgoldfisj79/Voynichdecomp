# MG3 scoring implementation amendment — 2026-10-03

Timing: written after MG3 jobs entered fitting, before any MG3 fit completion, generation output, or scientific statistic was observed.

The original mg3.py score implementation is computationally redundant: it reconstructs candidate edit-distance neighbourhoods separately for each of 32 generated replicates.

Scientific definitions, model, generated corpora, random seeds and decision rules are UNCHANGED.

Implementation amendment:
- precompute each outer fold's full raw-character Levenshtein matrix once with RapidFuzz, capped at ED=4;
- before scoring, verify 2,500 deterministic random pairs per fold (12,500 total per layer) against the frozen joint_model.distance_raw native implementation;
- require exact equality, symmetry and zero diagonal;
- then compute the preregistered folio_seen_Mplus and folio_seen_TTR from the cached matrices.
- All original run_joint diagnostics remain computed by the original frozen code.

This is a speed-only scorer replacement. Any ED verification mismatch aborts the job. No threshold or statistic is changed.
