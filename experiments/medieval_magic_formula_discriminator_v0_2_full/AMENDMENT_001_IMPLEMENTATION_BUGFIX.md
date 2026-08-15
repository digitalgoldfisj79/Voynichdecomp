# Amendment 001 — implementation bugfix

Date: 2026-08-15
Status: prospective with respect to v0.2 Voynich scoring.

During the first local execution attempt, the runner failed before producing any v0.2 external freeze and before acquiring or scoring Voynich data. The failure was a Python implementation error in four generator-control calls:

`str(rng.choice(Atrain)).text`

The selected object is already the expected record object, so wrapping it in `str(...)` removes the `.text` attribute and raises `AttributeError`.

The executable correction is:

`rng.choice(Atrain).text`

at all four affected call sites.

This amendment changes no source corpus, split, metric definition, generator family, parameter grid, nuisance control, permutation count, FDR rule, Voynich representation, or interpretation rule. It only makes the frozen v0.2 computation executable. `python -m py_compile` passes after the correction.

The corrected local runner began its external-only phase before any Voynich acquisition. The identical corrected payload is the one to be launched remotely for the complete network-dependent run.
