# f116v recto/show-through control v0.2

This stage follows the v0.1 TIFF preflight. It tests whether residual lower-page structure is independent of f116r show-through.

```bash
python -m pip install -r requirements.txt
python run_stage2.py --output results/f116v_recto_v0_2 --max-dim 2400 --bootstrap 24
```

The executable downloads and hashes the 46 raw f116v 16-bit TIFFs, acquires Yale's f116r high-resolution IIIF image, registers reflected/fluorescence/transmission families, fits a non-negative show-through model, reruns the residual detector and confirms surviving components against native TIFF tiles.

The absence of a matching f116r raw MSI cube is a frozen limitation. No generative model contributes to the verdict.
