# Formal morpholocal calibration v0.2 result

Date: 2026-07-15

## Execution provenance

- Hugging Face job: `6a5725bf85d9643ce16d378d`
- Hardware: `cpu-xl`, 24 spawned workers
- Container: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`
- Frozen seed: `8675309`
- Positives: 96
- Controls: 320
- Annealing steps: 400
- Restarts: 2
- Runtime: 500 seconds (505 seconds including scheduling)
- Numerical validation maximum absolute error: `4.71482053399086e-09` bits, below the frozen `1e-7` tolerance
- Result SHA-256: `c12c48d5585dd4efc5935d29ca2eae46df3c1dabd6475ed89ae6eb7a3c0b1705`

The run used the parameters and gates in `SPEC.md` and `FREEZE_RECORD.json`. The scientific design and thresholds were not changed after the formal run began. Execution-only fixes made before the successful job were documented in commit `06b124608e8c1a1253aea99a861ff3a7ebc66abc`.

## Formal result

Verdict: **FAIL_MORPHOLOCAL_CLASS_CALIBRATION**.

- Positive recovery: **34/96**; Wilson 90% interval `[0.2788822372, 0.4374457519]`
- Production-control false positives: **0/320**; Wilson 90% interval `[0, 0.0083839386]`
- Median mapping accuracy: **0.5833333333**
- Median null F1: **1.0**
- Selector recovery: **1.0**
- Structure recovery: **0.5729166667**

Control-family results were all 0/80 false positives, with Wilson 90% upper bound `0.0327129639` for each of `cell_markov`, `context_iid`, `copy_mutate`, and `permuted_cipher`.

## Gate decisions

Passed:

- overall false-positive upper bound <= 0.05
- every control-family upper bound <= 0.10
- median null F1 >= 0.75
- selector recovery >= 0.80
- numerical validation <= 1e-7 bits

Failed:

- overall positive lower bound >= 0.70
- every positive-stratum lower bound >= 0.50
- median mapping accuracy >= 0.60
- structure recovery >= 0.65

The weakest planted mechanisms were `frequency_weighted` homophone selection (1/24) and `sticky_line_reset` selection (2/24). `iid_uniform` recovered 18/24 and `cyclic` recovered 13/24. Other substantial failures included balanced external profile 7/32, global keys 13/48, adjacent-length selector 14/48, and unequal class sizes 15/48.

## Scientific conclusion

The estimator is highly conservative against the four tested production-only null families but lacks adequate sensitivity across the preregistered bounded morpho-local nomenclator class. This is a formal failure of the frozen v0.2 decoder and calibration gate.

It does not prove that no historical cipher mechanism could generate the Voynich manuscript. It does establish that this frozen decoder cannot reliably recover the class it was designed to test. Under the preregistered rule, no Voynich-manuscript inference or decoding run is authorised from v0.2.
