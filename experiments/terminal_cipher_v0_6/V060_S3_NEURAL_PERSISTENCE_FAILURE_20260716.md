# v0.6 Family S3 neural persistence failure — 2026-07-16

## Affected job

- Hugging Face job: `Digitalgoldfish79/6a58a22e85d9643ce16d5fc6`
- Hardware: `a100x4`
- Seed: `1731`
- Planned updates: `30,000`
- Effective batch: `32`
- Synthetic examples completed: `960,000`

## Scientific execution status

Training completed all 30,000 updates. The final logged rolling losses were:

- plaintext cross-entropy: `0.2639210177809`
- boundary binary cross-entropy: `0.6583997173309326`

The job failed only after training when rank 0 attempted to create the private dataset repository `Digitalgoldfish79/v060-terminal-checkpoints`. Hugging Face returned HTTP 403: the injected token authenticated as `Digitalgoldfish79` but did not have permission to create a dataset under that namespace.

The checkpoint had been saved only to the ephemeral job filesystem and was therefore lost when the container exited. The learning trace is valid provenance, but no model weights survive and no development evaluation can be performed from this run.

## Independent permission probe

A separate `cpu-basic` preflight job, `Digitalgoldfish79/6a58b5ebb1669a49bf0774c1`, reproduced the same 403 before any GPU rerun. This confirms that the failure is a Hub repository permission defect rather than a training or distributed-compute defect.

## Remediation frozen before rerun

1. No GPU rerun may start until `v060_hf_checkpoint_preflight.py` completes both repository access and a real probe upload using the exact injected `HF_TOKEN`.
2. The target dataset must already exist or the token must have dataset-creation rights.
3. The rerun must use `v060_family_s_neural_train_resilient.py`.
4. The resilient trainer performs a write probe before corpus loading or optimisation.
5. Recovery checkpoints are uploaded every 10,000 updates and the final upload is retried up to five times.
6. The synthetic generator, architecture, loss, update count, effective global batch and locked evaluation remain unchanged.

No locked test split or Voynich data was opened.