# v0.6 Family P — final amendment runtime failure and execution sharding

Date: 2026-07-16

Status: **INFRASTRUCTURE/RUNTIME FAILURE — SCIENTIFIC RESULT PENDING**

## Failed monolithic execution

Hugging Face job `6a588de285d9643ce16d5f4a` ran the frozen final permitted Family P solver `v060_family_p_coordinate_final.py` on `cpu-xl` with a declared timeout of 14,400 seconds.

At reconciliation:

- Hugging Face still reported `RUNNING` after 16,328 running seconds;
- the logs contained only dependency installation and `Cloning into '/tmp/v'...`;
- no `V060_P2_TRIAL` row had been emitted;
- no summary, scientific SHA-256 or recoverable output existed.

The job was cancelled. It is recorded as an infrastructure/runtime failure, not a Family P development result.

## Execution diagnosis

The original script executes all 16 deterministic trials in one container and only emits a row after an individual trial completes. A cheap `cpu-basic` execution smoke, job `6a58d03db1669a49bf0777e2`, proved that:

- the branch cloned correctly;
- dependencies and imports succeeded;
- the solver emitted deterministic trial rows;
- output hashing and persistence worked;
- smoke scientific SHA-256: `5bcec70b4d72d2549e3c88a11c62b6998f90a83586dbaa052316037c3f6506d8`.

The monolithic incident is therefore attributable to full-scale runtime and unrecoverable all-trials packaging rather than a checkout or import failure.

## Sharding action

Execution-only wrapper `v060_family_p_coordinate_shard.py` was committed at:

`5a80c33ad23e331195157a4b5eb710375be300c9`

The wrapper:

- imports the frozen `solve_trial` implementation directly;
- preserves every frozen search constant and deterministic seed;
- permits only the development split;
- selects exactly one of the original 16 `(mode, replicate)` trials;
- emits the trial row immediately;
- writes a self-hashed JSON result.

This changes only job granularity and recoverability. It does not alter the scientific algorithm and is not a second Family P amendment.

## Current benchmark

Single-trial benchmark job `6a58d06885d9643ce16d638d` is running on `cpu-xl` for `(mode=periodic, replicate=0)` with a 12-hour timeout. The remaining 15 shards are withheld until this benchmark proves that one frozen trial completes within the available runtime envelope.

No locked test or Voynich data has been opened.
