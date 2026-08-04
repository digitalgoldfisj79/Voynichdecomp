# DINOv3–CATMuS HTR execution audit

Date: 2026-08-04

## Repository state

- Repository: `digitalgoldfisj79/Voynichdecomp`
- Branch: `experiment/f116v-dinov3-catmus-htr-v0.1-20260804`
- Base: `d0b38d8f1cc07a91af8c3d2d47f040c600adf1e5`
- Final result commit before this audit: `93b0058268faed4527951ade4e5377d51d94a498`

## Frozen dependencies

- Dataset: `CATMuS/medieval`
- Dataset revision: `e11965909ba89dea89476f665fc4d8541b0bf7a1`
- Encoder: `facebook/dinov3-vits16-pretrain-lvd1689m`
- Encoder revision: `114c1379950215c8b35dfcd4e90a5c251dde0d32`
- PyTorch execution image: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`

## Jobs

### Discovery and schema

- `6a71e29a6b79c09949c21fdf`: verified that `HF_TOKEN` was injected into Jobs.
- `6a71e2ae6b79c09949c21fe5`: verified authenticated access to the gated DINOv3 checkpoint.
- `6a71e6a26b79c09949c2201c`: inspected CATMuS streaming schema and sample rows.
- `6a71e6de6b79c09949c2201e`: metadata scan; cancelled because image decoding made it low-value.

### Pure frozen-DINOv3 CTC

- `6a71e85fa00abefd4b291b6b`: syntax check.
- `6a71e8746b79c09949c2203f`: initial launch; cancelled because a 5,000-row shuffle buffer wasted GPU time during acquisition.
- `6a71e9196b79c09949c22046`: failed before model loading because the SHA bucket split found no eligible test shelfmark in the local 30,000-row stream.
- `6a71ea466b79c09949c22052`: corrected manuscript-disjoint preflight; completed with `DINOV3_CATMUS_PREFLIGHT_FAIL`.

Pure-model metrics:

- untrained test CER: 0.9690;
- trained test CER: 0.9607;
- blank prediction: `e`.

### Hybrid pixel + frozen DINOv3

- `6a71ebb06b79c09949c22060`: syntax check.
- `6a71ebd36b79c09949c22062`: matched two-arm comparison; completed with `HYBRID_DINOV3_CATMUS_PASS`.

Hybrid metrics:

- CNN-only test CER: 0.9656;
- CNN+DINOv3 test CER: 0.5952;
- absolute CER improvement: 0.3704;
- held-out exact line accuracy: 0.0000;
- fused blank prediction: `PH`.

### f116v application

- `6a71ed51a00abefd4b291bc9`: application syntax check.
- `6a71ed706b79c09949c22079`: deterministic fused-model retraining and f116v inference; completed.

The raw view-specific outputs diverged substantially. No complete line or local phrase was accepted as a transcription.

## Open corrections

1. The original hash-based split was replaced only after it proved impossible to populate a test set from the locally clustered stream. The corrected split permanently assigns each first-seen shelfmark to one partition and verifies set disjointness.
2. Pure pooled DINOv3 failed the preflight and was not applied to f116v.
3. The pixel+DINOv3 model passed the architecture-comparison gate, but its 0.595 held-out CER prevents transcription claims.
4. Checkpoint upload failed because the available Hugging Face token has gated-model read access but no model-repository creation permission.

## Terminal compute state

At closeout, all experimental jobs had reached `COMPLETED`, `ERROR`, or `CANCELLED`; a final Jobs query reported no running jobs.
