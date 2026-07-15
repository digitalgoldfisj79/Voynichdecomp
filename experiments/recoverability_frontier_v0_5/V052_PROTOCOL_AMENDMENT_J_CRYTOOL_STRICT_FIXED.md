# Recoverability frontier v0.5.2 — protocol amendment J: strict fixed inventory

Date: 2026-07-15

Status: fixed before execution.

## Correction to amendment I

Disabling rare-symbol mutation did not fully freeze the inventory because independent restarts still generated alternative bounded inventories. The 1,000,000 × 8, acceptance 0.05 run selected a result with only 92.5991% inventory overlap, despite the primary restart beginning at 95.8729%.

## Strict ablation

Every restart now begins from a random permutation of the exact inferred plaintext-label multiset. No restart may alter label counts, and rare-symbol mutation remains disabled.

Thus the only active search mechanism is the independently sourced CrypTool architecture of exhaustive pair sweeps, linear cooling and calibrated simulated-annealing acceptance.

Development configurations:

- 1,000,000 steps × 8 restarts, target acceptance 0.05 and 0.20;
- 3,000,000 × 12, target acceptance 0.05 and 0.20.

English 384-character development gate remains 70% mean recovery.
