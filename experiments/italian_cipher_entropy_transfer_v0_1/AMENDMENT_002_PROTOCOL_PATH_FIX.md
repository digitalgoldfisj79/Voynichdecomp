# Amendment 002 — protocol path repair only

Date: 2026-08-15
Status: implementation/packaging repair before any external scientific output and before any target access.

The first hosted workflow attempt (GitHub Actions run 31906048879) stopped in the reconstruction/hash step because `PROTOCOL_ENTROPY_TRANSFER_V01.md` was absent from the expected repository path. The runner payload hashes passed, but all firewall, source-download, entropy-transfer, and scientific-analysis steps were skipped.

Therefore the failed run exposed **no external scientific result and no target data**. This amendment restores the already-frozen protocol text at the expected path and broadens the workflow push trigger so repository-path repairs can launch a clean rerun. It changes no mechanism, parameter, source family, entropy metric, sanity gate, stability rule, seed, or target firewall.

The next run is the first scientifically eligible hosted run of v0.1.
