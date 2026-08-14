# Phase 7 target-free QA

This QA is defined before any Phase-7 target score is computed.

1. Verify `protocol.json` against `PROTOCOL_SHA256`.
2. Verify the exact Phase-4 `persistence_operators.py` Git blob `d6ad9687c7bd76226d37f53324a064909375cb66`.
3. Compile `reset_operators.py`, `design_utils.py`, and `run_phase7_remote.py`.
4. Run reset-operator self-tests: continuous fixed and geometric arms must be exact aliases of Phase-4 `FIXED_RUN3` and `MARKOV_M4`; line-reset sequences must factor exactly into independently seeded 10-token line processes.
5. Run synthetic adjudication self-tests for every preregistered verdict.
6. Do not restore or call the target scorer in the freeze job.
7. Do not launch Phase-7 scoring if the completed Phase-6 summary reports `replication_guard_pass=false`.

The workflow file is intentionally withheld until the Phase-6 replication guard is known.
