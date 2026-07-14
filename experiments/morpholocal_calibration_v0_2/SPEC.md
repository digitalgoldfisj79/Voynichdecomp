# Morpholocal calibration v0.2 — formal specification

The formal run is untouched by development outcomes except that the implementation is frozen after numerical GPU validation.

Design: 96 balanced synthetic positives and 320 matched controls; seed 8675309; 400 annealing steps; two restarts. Positive and control generators, scoring conventions, success criteria, and strata are those embedded in the hash-verified v0.2 source bundle.

Formal gates:

- overall positive Wilson 90% lower bound >= 0.70
- every positive-stratum Wilson 90% lower bound >= 0.50
- overall false-positive Wilson 90% upper bound <= 0.05
- every control-family Wilson 90% upper bound <= 0.10
- median mapping accuracy >= 0.60
- median null F1 >= 0.75
- selector recovery >= 0.80
- structure recovery >= 0.65
- GPU/scalar validation maximum absolute error <= 1e-7 bits

Any failed gate yields a formal unresolved verdict and prohibits manuscript analysis. No thresholds may be changed after the formal job begins.