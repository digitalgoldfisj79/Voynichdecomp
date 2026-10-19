# Conditional full-resolution GPU refinement

This stage is intentionally locked until the CPU preflight completes. It is not a second opportunity to tune thresholds on f116v.

## Registration

Use RoMa v2 on overlapping 1536-2048 px tiles for every accepted band against the frozen reference. Estimate local affine or homographic corrections with USAC/MAGSAC from high-certainty matches. Compare against the preregistered SIFT/ECC transforms. A correction is accepted only when it reduces held-out stable-edge reprojection error and does not increase blank-control candidate density.

## Dense representation

Extract DINOv3 patch features from the reference and the spectrally fused cube. Train only a linear, spatially grouped head on the same visible-stroke and parchment controls used in the preflight. The DINO branch is supporting evidence only when its candidate map independently overlaps the spectral candidate map.

## Restoration visualizations

Run DocRes appearance enhancement/deblurring and a fidelity-oriented HAT restoration on the surviving-text crops. A diffusion document-restoration model may be included as a third visualization. Every output must be accompanied by:

- the exact source crop;
- deterministic restoration;
- non-generative support overlay;
- edge-addition map showing pixels unsupported by any acquired band.

No restoration output contributes to the hidden-text verdict. It may improve human inspection only.

## Full-resolution gate

A preflight candidate survives only if it recurs at native resolution in at least two independent band subsets, under both classical and RoMa-v2 registration, and under the frozen spectral thresholds. Otherwise it is closed as a downsampling or registration artefact.
