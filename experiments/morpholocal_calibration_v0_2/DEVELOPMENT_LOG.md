# Development log

The frozen development run used seed 424242, 96 positives, 64 controls, 400 steps, and two restarts.

GPU validation errors were below 5e-9 bits. The GPU run completed all 160 trials and returned SHA-256 58dc06717ca8e408edb0fcf6cef25dc542ed3a26e4639b77dbfae13ba7723d2d.

Verdict: FAIL_MORPHOLOCAL_CLASS_CALIBRATION.

Results: positive recovery 32/96; false positives 0/64; median mapping accuracy 0.583333; median null F1 1.0; selector recovery 1.0; structure recovery 0.53125.

The CPU reference produced the same overall totals and verdict. It had median mapping accuracy 0.572917 and structure recovery 0.510417. The class-level conclusion is invariant across implementations.

The estimator is conservative against the tested nulls but insufficiently sensitive across the intended positive class. No manuscript inference is authorised from this result.