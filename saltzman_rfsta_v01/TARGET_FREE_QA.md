# Target-free QA

Completed before the workflow file is added.

- Protocol JSON frozen and SHA-256 recorded: `8952d3d7990a349e3f8b6cbce4786a69067ace95bed6e07ef935efb072741592`.
- New branch is descended from the completed Phase-9 branch but does not alter Phase 9.
- No `enriched_records.pkl` path exists in the corrected protocol or runner.
- Synthetic mechanism is not re-executed or re-optimised: the runner consumes the immutable Phase-8 COMBINED Q vectors.
- Canonical arm is fixed as `FIXED_LINE_RESET__POST`; continuous POST is sensitivity only.
- RF word segmentation is preserved when deriving aaa; the table's `~` is an analytical-unit separator, not permission to replace RF word boundaries.
- Non-clean/uncertain RF tokens and `<->` interruptions break adjacency rather than being silently concatenated.
- The workflow must verify all four prior source/conversion hashes before target scoring.
- The workflow must reconstruct and verify the unchanged scorer SHA-256 `926da655b603981bc197c248f6dce94fad7b242ab40a89d9d8d69cd40839d6b5` before target scoring.
- The target gate, 20 seeds, 200 null permutations, 80% stability rule, d3 gain rule, 0.15 match threshold and final adjudication labels are fixed in `protocol.json`.
- Any failure of the source hash, regenerated-aaa hash, scorer hash, target gate, or Phase-8 artifact guard stops or narrows inference; no automatic retuning is allowed.
