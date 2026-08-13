# Implementation fix 01 — pre-inference

The first v0.2 execution stopped during control `C0_N3`, before any ZLZI primary statistic or cross-frame hypothesis statistic was computed.

Cause: in the secondary line-length bookkeeping inside `analyse`, the expression `max(0, excess)` was applied to a NumPy array and raised `ValueError: truth value of an array is ambiguous`.

Fix: replace exactly that expression with `np.maximum(0, excess)`.

This changes no null, statistic, threshold, seed, hypothesis, eligibility rule, or decision criterion. All controls and analyses are rerun from scratch after the fix.
