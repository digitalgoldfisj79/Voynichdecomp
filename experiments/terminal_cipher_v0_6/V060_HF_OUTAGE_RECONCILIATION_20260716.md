# v0.6 Hugging Face outage reconciliation

Date: 2026-07-16

A Hugging Face service outage affected job visibility and GPU scheduling for approximately one hour. This note distinguishes valid scientific outputs from infrastructure failures. No threshold, seed, corpus split or scientific parameter was changed in response.

## Valid completed results retained

### Family S

- S1 true-codebook segmentation oracle: job `6a588d77b1669a49bf077023`; pass; SHA-256 `29aa04c41e3c01ac56712e98e6376f609d27405091548ca866200910d3c14cf1`.
- S2 initial segmented-key search: job `6a588eac85d9643ce16d5f4c`; failed reliability gate; SHA-256 `caad577a86baa793455b8d96d787e860b8f5bb88981c37cad975df64f5ec1ade`.
- S2 permitted restart escalation: job `6a588f4185d9643ce16d5f4e`; pass; SHA-256 `aae7812cf819e2014cfaf76b6e5ca072a5166d3ca022185cdc7c3141f9bf5486`.
- S3 classical blind SentencePiece segmentation: job `6a58920eb1669a49bf077096`; fail; SHA-256 `cf24de8ef8a52a21abf7527b1d1b0ab86bdcbe1df70b41b04e374ad107a7158a`.

### Family T

- T1/T2 component oracles: job `6a589123b1669a49bf077088`; both pass; SHA-256 `60db0399ec02b6abdf87a7cffcf8bfe4a75df1c75a6f0605e544bec2f881083b`.
- T3 initial blind joint solver: job `6a5893c485d9643ce16d5f68`; fail; SHA-256 `23d8b1c6df972946b2d150bc6f105c474dfcda1e5ce6fbcb195587a9a4dff906`.

### Family P

- P1 blind joint solver: job `6a588abdb1669a49bf076fe7`; fail; SHA-256 `4d2999963a1627257673f2e72d0f7cdfd46bd583f22ee45d1856550fafdc7725`.
- P2 permitted coordinate amendment: job `6a588de285d9643ce16d5f4a`; still reported running after service restoration. It remains authoritative unless it errors or reaches its frozen four-hour timeout.

## Invalid infrastructure runs

- S3 neural smoke job `6a58964385d9643ce16d5f6a` never cloned the repository because the selected PyTorch container lacked the `git` executable. The model, data and training loop were not reached. This is not a scientific result.
- Earlier compile or dependency failures remain excluded as documented in their logs.

## Recovery actions

- Added `v060_family_s_neural_smoke.py`, an execution-only wrapper that disables checkpoint upload for a five-update GPU smoke. It does not change model, synthetic generator, optimizer, losses or training updates.
- Submitted smoke jobs on smaller GPU flavours to validate the container path before allocating the frozen multi-GPU training run.
- Retained the original P2 process rather than duplicating or replacing it while it remains in a valid running state.

## Integrity statement

All valid results above were completed with preregistered data and parameters. Infrastructure relaunches reproduce the same deterministic configuration. No test split has been opened and no Voynich text has been scored.