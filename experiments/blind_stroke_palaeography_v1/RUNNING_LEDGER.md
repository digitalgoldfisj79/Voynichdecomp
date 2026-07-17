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

### Job `6a59dced85d9643ce16d74b9` — CANCELLED as an unbounded operational test

- Hardware: NVIDIA L4 ×1.
- The dataset mounted and package installation completed, but the original preflight recursively traversed approximately 1.7 million crop/proposal files before producing any checkpointed output.
- It was cancelled after roughly 35 minutes. No model-selection or Voynich analysis ran.
- Correction: replace only the access test with a bounded top-level inventory and explicit model probes. The frozen scientific protocol is unchanged.

### Job `6a59e14db1669a49bf079666` — FAILED source-transport smoke

- Failure: Base64 `Incorrect padding` while reconstructing the external-calibration source.
- Cause: the first large GitHub source field was truncated during transport.
- No calibration code ran.
- Correction: split both frozen sources into 4,000-character v2 parts and enforce exact byte-length and SHA-256 checks at reconstruction.

### Job `6a59e4e185d9643ce16d750a` — COMPLETED source and freeze smoke

- Reconstructed `external_calibration.py`: 36,922 bytes, SHA-256 `027e103fbcb440ec6614e535a8ab9d09f31ce3cee723a60c1ef6514aff0c1742`.
- Reconstructed `blind_model_selection.py`: 21,739 bytes, SHA-256 `c1597213530eb54cf3cd1093ab209dcecdce6fd8abf06362039f92480c523b9a`.
- All source files compiled.
- Frozen aggregate SHA-256: `78f57d1d1ea52c6a8a4f6de9438b094edc56b670ab22863767cfae659aaddeaa`.
- `phase1_voynich_opened=false`; `davis_labels_loaded=false`.

### Job `6a59e57285d9643ce16d7513` — FAILED before access probes

- The selected PyTorch GPU image did not contain `git`.
- No model or data test ran.
- Correction: fetch the single committed smoke-test file directly by immutable GitHub commit URL.

### Job `6a59e5ba85d9643ce16d7517` — MODEL AND READ-ACCESS PASS; WRITE-ACCESS FAIL

- Hardware: NVIDIA L4 ×1.
- Dataset mounted read-only with top-level entries `.gitattributes`, `crops`, `register`, `results`, and `src`.
- DINOv3-B/16 loaded and produced finite `[1, 201, 768]` hidden states.
- Historical TrOCR loaded and produced finite `[1, 577, 768]` encoder states.
- The OAuth credential successfully authenticated as `Digitalgoldfish79` but returned HTTP 403 when asked to create `Digitalgoldfish79/blind-scribal-hands-v1`.
- Root-level checks for the corpus manifest and embedding file were false because the live dataset stores them below the reported subdirectories; this is a path-audit issue, not a missing-mount conclusion.
- Davis data loaded: no.

### Job `6a59e60185d9643ce16d751d` — EXISTING HUB REPOSITORY WRITE FAIL

A direct commit probe to `Digitalgoldfish79/v060-terminal-checkpoints` also returned HTTP 403 and requested pull-request mode. The connected OAuth token therefore lacks direct repository commit scope; this is not a repository-name error.

### Job `6a59e627b1669a49bf0796bd` — WRITABLE VOLUME PROBE FAIL

The existing dataset volume mounted read-only. No pre-existing asset was changed.

### Result-transport amendment

Because compute/model/data reads work but Hub commits do not, external jobs use `run_external_calibration_logged.py`. It alters only transport: the frozen calibration code runs unchanged, excludes the large recomputable feature array, and emits a SHA-256-verified compressed bundle of JSON/CSV results to the job log. Those bundles are then committed through the authenticated GitHub connector. This does not modify folds, features, models, seeds, thresholds or selection logic.

### Job `6a59e6ca85d9643ce16d752d` — RUNNING external-control smoke

- Corpus: Historical-WI.
- Non-confirmatory smoke panel: 20 writers, two pages each, one fragment per page, two tiles, three permutations.
- Purpose: validate archive acquisition, filename parsing, preprocessing, both neural backbones, grouped evaluation and log-bundle persistence before full calibration.
- This run cannot unlock Voynich Phase I regardless of its metrics.

## Protocol freeze

`FROZEN_PROTOCOL.md` and `config/protocol_v1.json` were committed before any new Voynich model-selection result. The protocol fixes K=2–10, no K=5 preference, external controls, representations, fold-local nuisance removal, numeric calibration gates, discrete/continuous models, nulls, abstention and stopping rules.

`FREEZE_RECORD.json` records aggregate SHA-256 `78f57d1d1ea52c6a8a4f6de9438b094edc56b670ab22863767cfae659aaddeaa`.

Voynich Phase I remains unopened until every external-control gate passes.
