# ASC-LINE-SCALE-v0.1 target-free QA

Performed before the Phase-5 workflow existed and before any Phase-5 target scoring.

- Protocol SHA-256 recomputed locally: `74aaeffd009fbcebaa57327c2867ee15bf501385cdb822c9af9528a98c73a96a`.
- `design_utils.py` compiled successfully.
- `run_phase5_remote.py` compiled successfully without execution/import of the target scorer.
- Target-free design self-tests passed.
- Each frozen width `6, 8, 10, 12, 16, 20` retains exactly 2,000 tokens under the frozen grouping rule.
- OLS slope test recovered a known synthetic slope exactly.
- Synthetic adjudication tests recovered:
  - `ABSOLUTE_SHORT_SCALE_ROBUST` for positive per-width contrasts plus slope-equivalent families;
  - `SHORT_SCALE_ROBUST_WITH_LINE_INTERACTION` for positive per-width contrasts plus materially negative slopes;
  - `LINE_RELATIVE_SHIFT_SUPPORTED` when a large-width contrast breaks and both slopes are materially negative;
  - `P4_CURVE_NOT_REPLICATED_W10` when the replication guard is false.
- Phase-5 uses the unchanged Phase-4 persistence operator, Git blob `d6ad9687c7bd76226d37f53324a064909375cb66`.
- No `scorer.one_eval` call or Phase-5 target-derived statistic was run during protocol design or QA.

The workflow is to be committed only after this record and all preregistration/code files.
