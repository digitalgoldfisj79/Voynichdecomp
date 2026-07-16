# v0.6 Family S3 — seed 1731 complete; seed 1732 launched

Date: 2026-07-16

## Seed 1731

Hugging Face job: `Digitalgoldfish79/6a58e6ef85d9643ce16d6550`

Status: **COMPLETED**

- hardware: A100 x4;
- updates: 30,000;
- effective batch: 32;
- generated train examples: 960,000;
- runtime: 4,242 seconds;
- final checkpoint bytes: 102,032,251;
- final checkpoint SHA-256: `bf31c7ad18c65170d4525f834bb3b32d1a283dc1a693c4356f8c17eb4db6d206`;
- final manifest: `v060/s3/seed1731/u30000/s3_neural_seed1731_u30000.pt.manifest.json`;
- exact-byte roundtrip verified: true.

Recovery checkpoints at 10,000 and 20,000 updates completed before the run advanced and are present as three bounded shards plus manifest at their registered object prefixes.

No development evaluation, test split, Voynich data or Davis labels were opened.

## Companion seed clarification

The registered two-model ensemble omitted the numeric companion seed. Before any development evaluation, `V060_FAMILY_S_SECOND_SEED_EXECUTION_CLARIFICATION.md` fixed it as adjacent seed `1732`, without reference to recovery outcomes.

Exact seed-1732 signed-upload namespace preflight:

- job: `Digitalgoldfish79/6a58f93fb1669a49bf077d6a`;
- status: passed;
- fixed path: `v060/s3/preflight-seed1732/u30000/s3_neural_seed1732_u30000.pt.part-99999`;
- bytes round-tripped: 35.

The temporary self-test endpoint was removed before training launch.

## Seed 1732

Hugging Face job: `Digitalgoldfish79/6a58f9b185d9643ce16d676a`

Status at record creation: **RUNNING**

- hardware: A100 x4;
- updates: 30,000;
- effective batch: 32;
- seed: 1732;
- persistence checkpoints: 10,000, 20,000 and 30,000 updates;
- launch Git head: `2c812872d0e3582be3505956d9415a2aeb14af65`.

No S3 development evaluation is permitted until the seed-1732 final checkpoint completes exact-byte roundtrip verification.