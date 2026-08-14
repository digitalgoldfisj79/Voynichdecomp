# Phase 8 target-free QA

Defined before Phase-7 results.

1. Verify `protocol.json` against `PROTOCOL_SHA256`.
2. Verify exact Phase-4 persistence operator and Phase-7 reset operator sources.
3. Compile Phase-8 scorer and adjudication code before restoring the target scorer.
4. Synthetic self-tests must cover every preregistered ordering verdict.
5. For each document/replicate, PRE and POST must reuse the same cipher plan generated from untouched plaintext; add an assertion that no second plan is generated after the state operation.
6. Phase-8 scoring is forbidden until Phase 7 completes with `replication_guard_pass=true`.
7. Before ordering adjudication, every POST stratum must reproduce its Phase-7 median to `1e-12`; expected values are recorded with the immutable Phase-7 artifact id/digest.
8. The sealed Phase-9 consequence-panel checksum must remain unchanged.
