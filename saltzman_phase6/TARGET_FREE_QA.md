# Phase 6 target-free QA

Completed before the workflow file was added and before any Phase-6 target scorer was invoked.

- `protocol.json` canonical SHA-256: `5a75df7cdc08dcb6a7c078801aabdce9e4d939b712a0a4460987624a18610710`.
- `dwell_operators.py`, `design_utils.py`, and `run_phase6_remote.py` compile successfully.
- `design_utils.py` synthetic adjudication tests pass for equivalence, tau3-only gradient, non-monotonic law effect, and replication failure.
- Semi-renewal tau=3 exact moments: E[D]=8/3, E[D^2]=8, Var[D]=8/9, hence E[D^2]/E[D]=3.
- Semi-renewal tau=4 exact moments: E[D]=15/4, E[D^2]=15, Var[D]=15/16, hence E[D^2]/E[D]=4.
- Fixed tau3/tau4 operators are exact aliases of Phase-4 RUN3/RUN4 state construction.
- Geometric tau3/tau4 operators are exact aliases of Phase-4 MARKOV_M4/M5 state construction.
- Long-sequence target-free sanity checks confirm all six state laws are binary, short-correlated, and decay by lag 12.
- No Phase-6 call to `scorer.one_eval` or any target-derived scoring occurred during QA.

The workflow must independently repeat the protocol/hash/operator checks before restoring the scorer payload.
