# Post-target fix 01 — sparse LLCT bootstrap

Frozen: 2026-08-13, after the LLCT primary statistics and controls had been printed, but before any successful H2/H3 bootstrap result.

The LLCT primary run qualified all controls and printed the full F/R target profile. It then failed in H2/H3 bootstrap evaluation because at least one charter-block resample contained zero observed exact repetitions for a profile component. The preregistered H2 distance is defined in log-ratio space, so a ratio of zero has no finite logarithm.

Because this issue was discovered **after target inspection**, no continuity correction, pseudocount, alternative distance, feature removal, or resampling unit is introduced.

The completion wrapper `run_llct_formulaic_profile_completion.py` therefore applies the most conservative handling:

- all original controls, primary calculations, seeds, and thresholds are unchanged;
- H3 is evaluated independently with its preregistered ED1 log-ratio contrast; no correction is needed unless an ED1 bootstrap ratio itself is zero;
- H2 finite bootstrap replicates are summarized descriptively;
- if even one of the requested 1,000 H2 bootstrap replicates has an undefined log-ratio component, the **literal H2 verdict is `UNRESOLVED_ZERO_BOOTSTRAP`** regardless of what the finite subset suggests;
- no post-target modification may turn H2 into a positive preregistered result.

The already-exposed LLCT primary values are retained as valid descriptive measurements because the failure occurred only after their computation and all preregistered controls passed.