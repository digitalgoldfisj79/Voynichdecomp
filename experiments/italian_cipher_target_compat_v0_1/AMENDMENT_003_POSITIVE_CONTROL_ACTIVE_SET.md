# Target Compatibility Amendment 003 — positive-control active set

Date: 2026-08-15
Status: **pre-target amendment**. No target entropy values have been computed in this stage.

A final hostile preflight identified a logical loophole in Amendment 002. Its global `Delta_any` minimum ranged over every individually qualified mechanism, even if the mechanism's parent group failed the external positive-control gate. In principle, an externally non-discriminating group could therefore create the target's global improvement while a different group merely satisfied the compatibility condition.

This amendment closes that loophole prospectively.

## Active mechanism set
After external leave-one-source-family-out positive-control qualification, define:

`ACTIVE = {m : m is individually qualified AND group(m) passed its external positive-control gate}`

where the primary groups are HOM, NOM, and BIGRAM.

The global statistic is henceforth:

`Delta_any = d_ID - min(d_m for m in ACTIVE)`.

The external plaintext null uses this exact same ACTIVE set, which is determined entirely from external generated controls before target access.

If `ACTIVE` is empty, external calibration fails and the target is not downloaded.

## Candidate condition
A target can reach `HISTORICAL_CIPHER_ENTROPY_CANDIDATE` only if:

1. target median `Delta_any` exceeds the externally frozen global plaintext-null threshold built from ACTIVE;
2. target median distance is within the externally frozen LOSO compatibility threshold of at least one mechanism in ACTIVE.

This supersedes any wording in Amendment 002 that allowed a mechanism from a group failing positive-control qualification to contribute to the global minimum.

No target value, representation, source family, historical parameter, or entropy threshold informed this amendment.
