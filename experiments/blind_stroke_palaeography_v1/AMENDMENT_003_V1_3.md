# AMENDMENT 003 — Preregister external calibration v1.3 with local DINOv3 ViT-7B/16

**Date:** 2026-07-17  
**Status:** new preregistered version; prospective backbone substitution before calibration metrics  
**Voynich Phase I opened:** no  
**Davis labels loaded:** no  
**f115r target boundary loaded:** no

## Trigger

External-calibration v1.2 could not reach DINO feature extraction because the frozen checkpoint `facebook/dinov3-vitb16-pretrain-lvd1689m` is gated and unavailable to the authenticated Hugging Face account. The terminal end-to-end smoke error was HTTP 401 / `GatedRepoError` while requesting the DINOv3 processor configuration.

No DINO embedding, HTR embedding, retrieval score, nuisance comparison, selected representation, permutation statistic, synthetic-K statistic, Voynich partition or Davis comparison was produced under v1.2.

The user supplied a private Hugging Face bucket containing a complete DINOv3 ViT-7B/16 checkpoint:

`Digitalgoldfish79/dinov3-vit7b16-pretrain-lvd1689m-bucket`

The bucket audit established:

- architecture `DINOv3ViTModel`;
- hidden size 4096;
- 40 transformer layers;
- patch size 16;
- six safetensors weight shards;
- total required model bytes 26,864,214,342.

## Model-only preflight

A model-only A100 preflight copied the bucket checkpoint to local NVMe, verified each required file by byte size and SHA-256, loaded all six local shards, and ran one 224×224 image through the model.

Observed preflight facts:

- local checkpoint-shard loading: 7.02 seconds;
- allocated GPU memory after loading: 12.51 GiB;
- output shape: `[1, 201, 4096]`;
- output finite: true.

This preflight did not use Historical-WI writer labels or generate any calibration-performance statistic.

## v1.3 preregistered changes

Version 1.3 retains the complete v1.2 corpus and fold specification and makes the following prospective substitutions:

1. Replace inaccessible DINOv3 ViT-B/16 with the supplied DINOv3 ViT-7B/16 checkpoint.
2. Copy the checkpoint from the read-only bucket mount to job-local NVMe before `from_pretrained`, because direct safetensors loading through the remote mount stalled at the first shard.
3. Verify the required local checkpoint files against the frozen byte sizes and SHA-256 digests recorded in `code/uv_v1_3_smoke.py`.
4. Load DINOv3 ViT-7B/16 in bfloat16 and cast input pixels to the loaded backbone parameter dtype. The historical-TrOCR model remains float16 and otherwise unchanged.
5. Record the DINO model identifier in outputs as `Digitalgoldfish79/dinov3-vit7b16-pretrain-lvd1689m-bucket`.
6. Use DINO batch size 4 for the end-to-end smoke. Batch size is an execution parameter only and does not alter images, representations, folds, labels, metrics, selection or thresholds.

## Elements unchanged from v1.2

- Historical-WI uses all three physical pages per selected writer.
- Colour and binarized derivatives share the same physical-page identifier; the colour derivative is the independent retrieval item when both exist.
- Three deterministic writer-balanced page-group folds are used for Historical-WI.
- Same-page gallery exclusion remains mandatory.
- Classical, DINO, HTR, family-residual and frozen ensemble definitions are unchanged.
- Fold-local nuisance removal and family residualization are unchanged.
- Writer-selection seed, panel construction, permutation method and synthetic-K method are unchanged.
- All confirmation thresholds and abstention requirements are unchanged.
- Voynich Phase I remains unopened until the complete external-control gate passes.

The larger DINO backbone changes representation capacity and therefore requires this new preregistered version. It is not treated as an implementation-only repair.

## Immutable derivation and runner

Runner:

`code/uv_v1_3_smoke.py`

Runner commit:

`8d46a5fdd192e226f5b79c67d108147ef4a3cbb5`

Runner bytes and SHA-256:

- bytes: `7800`
- SHA-256: `2dc22a86e4cbaffca51f2714ca04a77578c2c56e759861e15d084e46dc84eb2a`

The runner imports the immutable v1.2 smoke source from commit `edf418ff10a8f26db6e01fb5fcd1a5b68f0046e5`; v1.2 itself reconstructs and verifies the frozen v1.1 parent source before applying its preregistered substitutions.

The resulting v1.3 calibration source was derived before execution with:

- bytes: `38153`
- SHA-256: `0415c0f0e3fd5e70c553055b29ee32c8b0d45a3de5ccbbd69bb2d8687a77371f`

Any mismatch in source or checkpoint verification must stop the run before metrics.
