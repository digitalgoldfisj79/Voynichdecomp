# VBM v141 admission package

This is **not a new VBM cipher, decipherment, or qualified Voynich generator**. The frozen v138 architecture failed the necessary feasibility checks; our agreed NO-GO rule therefore stopped further development and inferential experiments.

## Reproduction

You need the source file `Generator_v138_Pruefpaket_2026-10-14.zip` beside this folder or specify its absolute path.

```bash
python -m pip install networkx numpy scipy
python preflight.py /path/to/Generator_v138_Pruefpaket_2026-10-14.zip
python adversarial_review.py
```

The scripts recreate a local `source/` copy from the *original ZIP*. They run the unchanged v138 generator in isolation, check it reproduces the reference, calculate constraint clashes, optimise cell assignments with min-cost flow, and independently verify the optimum with Hungarian assignment. Checkpoints use atomic `.pkl` replacement. No original package is modified.

Read `VBM_v141_Admission_Report_2026-10-14.md` (starting with the retractions), `admission_contract.json`, `results.json`, and `independent_review.json`.

## Limitations

This demonstrates a failure of the supplied 443-cell one-cluster-one-cell architecture, *not* an impossibility proof for any conceivable VBM mechanism. The best reassignment is a necessary-condition upper bound, not measured decoding accuracy. No Voynich held-out fitting/test or new decoder was run, because the admission contract did not qualify the existing architecture.
