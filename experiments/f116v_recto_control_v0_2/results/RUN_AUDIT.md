# Stage-2 execution audit

Date: 2026-08-04

## Final controlled run

- Hugging Face Job: `6a71a0b0a00abefd4b29116f`
- Hardware: `cpu-xl`
- Status: `COMPLETED`
- Runtime: 211 seconds
- Frozen code commit: `5f01df206b0f860a6cb1353db9f5d366a13f1ea0`
- Analysis resolution cap: 2400 pixels
- Bootstrap repetitions: 24
- Source TIFFs: 46 raw 16-bit f116v captures

## Corrections made during execution

1. OpenCV 5 required explicit float32 normalization before Sobel registration.
2. The first completed scientific run exposed an invalid negative-decision path: the synthetic positive control had failed but the code still emitted a negative verdict.
3. The synthetic control was corrected to use a signed acquired spectral signature and a guaranteed eligible placement, with an attenuation ladder.
4. The verdict gate was corrected so failed sensitivity or recto specificity forces `RECTO_CONTROL_INCONCLUSIVE`.
5. Repeated gdown access triggered public-link throttling. Direct `drive.usercontent.google.com` downloads were substituted without changing source bytes.

All superseded or throttled jobs were allowed to terminate or were cancelled immediately; no orphaned paid compute remains active from this programme.
