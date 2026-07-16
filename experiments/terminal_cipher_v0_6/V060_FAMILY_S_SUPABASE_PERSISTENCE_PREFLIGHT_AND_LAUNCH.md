# v0.6 Family S3 — exact checkpoint persistence preflight and A100 launch

Date: 2026-07-16

Status: **PERSISTENCE PREFLIGHT PASSED; FROZEN SEED-1731 TRAINING LAUNCHED.**

No test data, Voynich text, Davis labels, scientific thresholds, model architecture, generator, optimiser, update schedule, loss, seed or selection rule was changed.

## Reason for execution-layer replacement

The Hugging Face MCP session can create and manage Jobs, but the OAuth credential delegated into a child Job returned HTTP 403 for dataset-repository commits. The original 30,000-update seed-1731 run therefore completed training but lost its local checkpoint at final upload.

The replacement changes only durable storage. It preserves the exact byte stream produced by `torch.save`.

## Frozen transport

- private Supabase bucket: `voynich-compute` in the existing `research-agent` project;
- exact checkpoint byte stream split into 40 MiB parts;
- each part stored without tensor conversion, quantisation or lossy compression;
- manifest records original filename, total byte count, whole-file SHA-256, and per-part byte counts and SHA-256 values;
- every uploaded part is downloaded through a fresh exact-path signed URL;
- parts are reassembled and the whole-file byte count and SHA-256 must match before the checkpoint is accepted;
- the uploader raises an error and stops training at the checkpoint barrier if verification fails.

Implementation commits:

- exact-byte transport: `8063817cf1a9f2fb67db85075c304b6c917c5e6d`;
- frozen trainer with 10k/20k/30k persistence: `a50ce9a0746c4994fbfe41ae2332a1af0882a650`;
- exact checkpoint preflight: `1a8621aaff1a3ad0cbff9593109d45a3ac9bcf3b`;
- signed-URL transport refinement removing all child-job credentials: `3612730fc6ea38c8ba3f42c1c24eb5d73ab7a7a2`.

## Exact checkpoint preflight

Hugging Face job: `Digitalgoldfish79/6a58e666b1669a49bf077a99`

Result marker: `V060_S3_EXACT_CHECKPOINT_ROUNDTRIP_PASS`

- checkpoint filename: `s3_neural_seed1731_u30000.pt`;
- checkpoint bytes: `102032251`;
- checkpoint SHA-256: `e5992baaf64672e31eb6c27e3be5327c85e7e51b75de9e7c536e4d10b3a8cc71`;
- shard count: `3`;
- manifest: `v060/s3/preflight-seed1731/u30000/s3_neural_seed1731_u30000.pt.manifest.json`;
- upload, download, reassembly and whole-file SHA verification: **PASS**.

This preflight constructs the actual frozen Transformer architecture and the exact final checkpoint payload shape; it is not a small placeholder-file test.

## Frozen training launch

Hugging Face job: `Digitalgoldfish79/6a58e6ef85d9643ce16d6550`

Hardware: `a100x4`

Frozen command parameters:

- seed: `1731`;
- updates: `30000`;
- batch per rank: `8`;
- world size: `4`;
- effective batch: `32`;
- warmup: `2000`;
- checkpoint updates: `10000`, `20000`, `30000`;
- final filename: `s3_neural_seed1731_u30000.pt`.

The run remains a development-only synthetic training run. Evaluation remains prohibited until the required two frozen models are present and verified.
