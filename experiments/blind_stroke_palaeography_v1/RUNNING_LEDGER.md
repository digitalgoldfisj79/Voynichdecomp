# Running ledger — blind stroke-level palaeography v1

All timestamps UTC. This ledger is append-only in scientific substance; later corrections do not erase earlier failures.

## 2026-07-17 — programme start

The governing fresh-chat charter was read and adopted. This programme is separate from the cryptanalysis/Cryptologia work. Global whole-word and connected-component K-means are prohibited. Davis assignments and the f115r target boundary remain unavailable to Phase-I modelling.

## Repository preflight

- Repository: `digitalgoldfisj79/Voynichdecomp`
- Authenticated permission: admin/push confirmed.
- Current `main` head when inspected: `b877744c3f0afcee28401e6c589c4c4daf07e542`.
- The current terminal-experiment lineage was found on `experiment/terminal-cipher-programme-v0.6-20260716`, 307 commits ahead of and 7 commits behind `main` at comparison time.

### Setup error retained

An initial branch, `experiment/blind-stroke-palaeography-v1-20260717`, was created from `main` before the branch divergence was visible. No scientific files were written to it. It is retained as provenance and is not the working branch.

### Correct working branch

`experiment/blind-stroke-palaeography-v1-20260717-full` was created from `experiment/terminal-cipher-programme-v0.6-20260716`. Prior experimental assets remain untouched.

## File-library audit

Recovered directly:

- DINOv3 implementation report, protocol, environment and prototype results;
- Nomic completion and negative-results methods notes;
- Davis 2020 paper and existing complete-read summary;
- checksums for the prior bundles.

Only checksum stubs, not the original ZIP bytes, were found for:

- `voynich_blind_hands_final.zip`: `9d5beb925a31d1fe93af47d0412562eaac34b88e8ce0e3e30bee3427ded36188`;
- `voynich_next_tests_complete.zip`: `5820d53e6d0119fa1fafb290e8fc73e4baf3a13e405048758eb9a4794be015ad`;
- `voynich_dinov3_baseline_complete.zip`: `02331cc3b8eb46c13bd158a5795ff92f1032dd98c3093a7cc5b24e8700f86dbd`.

The live Hugging Face crop dataset and repository lineage are therefore being audited as the operational source. This does not assert byte-equivalence to absent ZIPs.

## Compute smoke tests

### Job `6a59dbf485d9643ce16d74aa` — FAILED before scientific execution

- Hardware: NVIDIA L4 ×1.
- Intended purpose: mount the crop dataset, load DINOv3 and historical TrOCR, and verify Hub write access.
- Failure: `/bin/sh: set: Illegal option -o pipefail`.
- Interpretation: launcher-shell defect only. No Python scientific code ran and no result was observed.
- Bounded correction: relaunch the identical Python preflight under `/bin/bash`.

### Job `6a59dced85d9643ce16d74b9` — RUNNING at ledger creation

- Hardware: NVIDIA L4 ×1.
- Dataset mounted read-only: `Digitalgoldfish79/vdino3-crops` at `/vdino3`.
- Models requested: official gated DINOv3-B/16 and `Riksarkivet/trocr-base-handwritten-hist-swe-2`.
- Write target: new private dataset `Digitalgoldfish79/blind-scribal-hands-v1`, with an existing-repository fallback.
- Davis data loaded: no.

## Protocol freeze

`FROZEN_PROTOCOL.md` and `config/protocol_v1.json` were committed before any new Voynich model-selection result. The protocol fixes K=2–10, no K=5 preference, external controls, representations, fold-local nuisance removal, numeric calibration gates, discrete/continuous models, nulls, abstention and stopping rules.

Voynich Phase I remains unopened until the external-control gates pass.
