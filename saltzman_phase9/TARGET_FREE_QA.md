# Phase 9 target-free QA

1. Verify the Phase-9 protocol SHA, sealed 24-metric panel SHA, and final-pipeline selection-rule SHA.
2. Verify Git blobs for the existing 85-metric code (`3d1846b6ca469029b96067e87e4db98f877bb2e2`) and enriched Voynich records (`512dd1748ddf50f17aa53dccb685870f221877a9`).
3. Verify the immutable Phase-8 final reference has `replication_guard_pass=true` before any synthetic Phase-9 scoring.
4. Compile the Phase-9 runner and confirm exactly 24 unique confirmatory metrics and six four-metric families.
5. Build the Voynich baseline before synthetic scoring from immutable records/metric code. Confirm reconstructed line order flattens exactly to record token order.
6. Do not call the Saltzman Q scorer (`one_eval`, Q target, ED1/E1 gates) anywhere in Phase 9. The frozen scorer payload may be restored only to reuse its deterministic `seed_of` helper; no Q statistic is computed.
7. Use the same metric RNG seed for cipher-only/final/sensitivity arms within each production replicate.
8. Require every scoring shard to produce a non-empty JSON and use `set -o pipefail` wherever output is piped through `tee`.
9. Sensitivity pipelines are predeclared by the upstream selection rule and cannot replace the canonical primary pipeline on Phase-9 performance.
