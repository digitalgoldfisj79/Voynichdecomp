# v0.6 neural S3 — existing-repository persistence preflight result

Date: 2026-07-16

Status: **FAILED — A100×4 RETRAINING NOT AUTHORISED**

## Purpose

The original 30,000-update neural S3 training completed successfully but lost its checkpoint after a final Hub persistence failure. The hardened rerun is therefore conditional on a cheap CPU-only proof that the currently injected Hugging Face credential can persist a file to the already-created dataset repository:

`Digitalgoldfish79/v060-terminal-checkpoints`

No GPU retraining is permitted until this preflight passes.

## Provenance

- Repository: `digitalgoldfisj79/Voynichdecomp`
- Branch: `experiment/terminal-cipher-programme-v0.6-20260716`
- Existing-repository preflight commit: `731afcab9d441b9961ef317b6ca17f385d4675a0`
- Preflight script: `v060_hf_existing_repo_upload_preflight.py`
- Hardened trainer commit: `0b5d46d35546bcd47131fad49d57e57d58ccdd93`
- Target dataset: `Digitalgoldfish79/v060-terminal-checkpoints`

## Execution results

### Secret-forwarding syntax check

Job `6a58d225b1669a49bf07780d`, `cpu-basic`:

- `${HF_TOKEN}` was injected successfully.
- The job printed `V060_SECRET_PRESENT True`.

This resolves the earlier connector-layer block caused by the alternative `$HF_TOKEN` argument form.

### Direct existing-repository upload

Job `6a58d235b1669a49bf077813`, `cpu-basic`:

- repository clone and script execution began normally;
- the credential authenticated and could inspect the existing dataset;
- the upload failed at the Hub commit endpoint with HTTP 403;
- Hub response: direct commit to `main` was forbidden and suggested `create_pr=1`.

### Pull-request upload fallback

Job `6a58d262b1669a49bf07781f`, `cpu-basic`:

- upload was retried with `create_pr=True`;
- the request failed at the Hub pre-upload endpoint with HTTP 403 `Authorization error`;
- therefore the credential cannot persist through a dataset pull request either.

### Credential identity

Job `6a58d279b1669a49bf077821`, `cpu-basic`:

- identity: `Digitalgoldfish79`;
- authentication type: OAuth;
- reported expiry: `2026-07-16T19:48:40.000Z`.

The failure is therefore not absence of a credential or an incorrect identity. The injected OAuth grant lacks the repository write permission required by the Hub content endpoints.

### Dataset-volume fallback

- Job `6a58d1e285d9643ce16d63aa` mounted the dataset successfully and listed `.gitattributes`.
- Job `6a58d20585d9643ce16d63ac` proved the mount is read-only: creation of `preflight/v060_mounted_dataset_write_probe.json` failed with `Read-only file system`.
- An explicit read-write dataset mount was rejected by the Hugging Face Jobs API because model, dataset and Space volumes are read-only only.

## Decision

The mandatory persistence preflight fails. The hardened A100×4 retraining is not launched, because neither the 10,000-, 20,000- nor 30,000-update checkpoint could be guaranteed to survive container exit.

This is an infrastructure/credential-scope failure, not a neural-model development result. The frozen neural architecture, training schedule and evaluation remain unchanged. Neural S3 remains pending until a write-capable Hugging Face credential can pass the same CPU-only existing-repository upload probe.
