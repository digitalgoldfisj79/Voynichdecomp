# Yiddish Cipher Control v1 — statistical bound addendum

Status: **FROZEN BEFORE ANY FULL QUALIFICATION RUN**  
Date: 2026-09-12

This addendum corrects a sample-size inconsistency found during preflight review of `PROTOCOL.md`. No full-profile result had been exposed when this correction was made. The smoke profile is explicitly non-qualifying and therefore cannot consume a scientific confirmation condition.

## C4 family-wise false-positive bound

The base protocol requires the upper one-sided bound on the false-positive probability to be <=0.05. A panel of only 32 trials cannot establish that bound even with zero false calls: the one-sided 95% Clopper-Pearson upper bound is about 0.089.

For the full control we therefore use the more conservative multiplicity rule already used elsewhere in the Yiddish qualification programme:

- six registered negative/comparator families;
- family-wise Bonferroni alpha = 0.05 / 6;
- exact one-sided Clopper-Pearson upper bounds;
- **minimum n=94 trials per gated negative family**.

With 0/94 false-positive calls, the upper bound is approximately 0.0496555. Any false calls are evaluated using the same exact bound; no normal approximation is used.

The smoke profile may use smaller n solely to establish that the code path executes. It always returns a non-qualifying state.

## C5 metamorphic replication

Every stochastic metamorphic relation that depends on solver search rather than an algebraic identity must be run on at least 32 planted instances in the full profile. In particular, MR1 global ciphertext-symbol relabelling uses 32 paired solver runs in the qualifying profile. Deterministic algebraic identities such as exact duplication/recombination of sufficient statistics do not acquire evidential value from artificial repetition and may be checked once exactly.

## Interpretation

This is a preregistered repair of the **control instrument**, not a change made in response to Voynich or to a full qualification outcome. Voynich remains sealed. The addendum supersedes any conflicting full-profile sample counts in the initial executable implementation.