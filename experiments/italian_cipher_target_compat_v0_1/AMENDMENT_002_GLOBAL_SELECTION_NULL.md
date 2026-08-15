# Target Compatibility Amendment 002 — global mechanism-selection null

Date: 2026-08-15
Status: **pre-target amendment**. No target entropy values have been computed in this stage.

A hostile review of the frozen target protocol and Amendment 001 identified a remaining multiplicity problem. The planned target analysis would inspect three primary mechanism groups (`HOM`, `NOM`, `BIGRAM`) and could therefore obtain an apparently favourable result merely by taking the best of several externally qualified alternatives. Calibrating a separate plaintext null for each group does not control that higher-level selection.

This amendment freezes one global primary statistic before target scoring.

## Global primary statistic
For every observed 1200-symbol window define

`Delta_any = d_ID - min(d_m)`

where the minimum ranges over **all externally qualified individual mechanisms** in the primary pre-1450 groups:

- HOM: `HOM2`, `HOM34`, `HOM34_NULL_1`, `HOM34_NULL_2p5`, `HOM34_NULL_5`
- NOM: `NOM25`, `NOM50`, `NOM100`
- BIGRAM: `BIGRAM10`, `BIGRAM20`, `BIGRAM40`

Diglyph and combined mechanisms are excluded from positive primary selection because the external parent run found rendering-sensitive H1 behaviour. `ORTHO_POST1460` is excluded because it is chronologically secondary.

## External global plaintext null
For each leave-one-source-family-out fold:

1. fit every externally qualified primary mechanism and the identity model using the other five source families under Amendment 001's family-balanced H0/H1 calibration;
2. score the held-out **identity/plaintext** windows;
3. calculate `Delta_any` for each held-out plaintext window;
4. record the median `Delta_any` for that held-out source family.

The global plaintext-null threshold is the **maximum of the six held-out-family medians**.

This exactly mirrors the target operation while giving each external source family one vote.

## Target verdict supersession
Group-specific `Delta_group` values remain diagnostics only. They cannot by themselves yield a positive primary verdict.

A target may reach `HISTORICAL_CIPHER_ENTROPY_CANDIDATE` only when all of the following hold:

1. target median `Delta_any` exceeds the frozen global external plaintext-null threshold;
2. at least one primary mechanism group passed its external positive-control qualification;
3. the target median distance is within the externally frozen LOSO compatibility threshold of at least one individual mechanism belonging to a positively qualified primary group.

If condition 1 fails, verdict is `NO_CIPHER_ENTROPY_ADVANTAGE`.

If condition 1 passes but condition 3 fails, verdict is `ENTROPY_ADVANTAGE_BUT_OUT_OF_DISTRIBUTION`.

## Circularity firewall
This correction was made before target entropy scoring. It introduces no target value, source-language choice, representation change, historical parameter change, or target-dependent threshold. Its sole purpose is to control the multiplicity created by selecting the most favourable historically plausible primary mechanism.
