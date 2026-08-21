# SVT v0.5.0 — Targeted Latin portability gate

Status: **FROZEN BEFORE BINDING EXECUTION**

Purpose: test whether the two SVT components already qualified on German are portable to Latin without any joint-decoder tuning.

Latin source: UniversalDependencies/UD_Latin-PROIEL, commit `bc36b0223deeaa86d1a5aa48d464770863c0fc7b`.
Pinned git blob SHA-1: train `1a02fc3f95f9a2d64249dbadb6877706759a96d5`; dev `e9857ad6d660c34329ddfbd59d4a1037665603e9`; test `d32ce9d3d3c3bcb149166e53e0a38476ec3afaa8`.
Normalization and character-level language modelling follow the existing recoverability harness.

## L1 — fixed-boundary state/key portability
- 8 fresh Latin synthetic trials: periodic/line_reset x replicates 0..3.
- plaintext length 1536; namespace offset 31000.
- true cipher-unit boundaries supplied.
- hidden: mode, period, key, plaintext.
- same 22 candidate structures, top-6 screen, 12-start qualified key solver, and primitive-period canonicalisation as v0.3.4.
- PASS unchanged from v0.3.4 ordinary arm: exact canonical structure 8/8; mean recovery >=0.95; median >=0.97; minimum >=0.85; 8/8 recovery >=0.90.

## L2 — standalone segmentation portability
- 8 fresh Latin synthetic trials: periodic/line_reset x replicates 0..3.
- plaintext length 1536; namespace offset 33000.
- input: unsegmented verbose surface and observed line starts only.
- same v0.4 semi-Markov segmenter; no state/key or plaintext-language help.
- PASS unchanged from v0.4 S0: mean boundary F1 >=0.90; median >=0.90; minimum >=0.85; 8/8 >=0.85; mean absolute unit-count relative error <=0.05.

Overall PASS requires L1 and L2 both PASS. No v0.4.x joint decoder is run. No post-hoc threshold changes, Latin-specific parameter tuning, shortlist widening, or namespace reuse. Voynich remains SEALED.
