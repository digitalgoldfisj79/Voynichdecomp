# Recoverability frontier v0.5.2 — protocol amendment I: fixed-inventory CrypTool ablation

Date: 2026-07-15

Status: fixed after the first two CrypTool-port development configurations, before this ablation.

## Evidence

The English inferred inventory begins at 95.8729% mean overlap with the true observed inventory.

Completed CrypTool-style configurations:

- 1,000,000 steps × 8 restarts, target acceptance 0.05: 24.5117% recovery, final inventory overlap 89.0558%, mean 133 mutation events;
- 1,000,000 × 8, target acceptance 0.20: 10.5794% recovery, final overlap 84.5621%, mean 117.875 mutation events.

The rare-symbol mutation mechanism is therefore counterproductive in this bounded benchmark, where a strong inventory estimate already exists.

## Frozen ablation

Retain the independently sourced CrypTool search elements:

- exhaustive pair sweeps;
- linear cooling;
- calibrated initial temperature;
- multiple independent restarts;
- acceptance-probability floor `0.0085`.

Disable rare-symbol inventory mutation. The inferred inventory in the primary restart is preserved. Random bounded inventories remain available in independent restarts, but the global best can never be worse than the primary fixed-inventory trajectory.

Development configurations:

- 1,000,000 steps × 8 restarts, target acceptance 0.05 and 0.20;
- 3,000,000 × 12, target acceptance 0.05 and 0.20.

English 384-character development gate remains 70% mean recovery.
