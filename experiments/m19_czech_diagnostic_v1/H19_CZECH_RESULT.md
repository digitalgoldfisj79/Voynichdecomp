# M19 Czech Diagnostic v1.1 — H19 STA-Family Result

Date: 2026-08-12
Status: **CLOSED AT H19 FAMILY — NO CZECH SIGNAL**
Binding protocol: `PROTOCOL_V1_1.md`
Scientific runner freeze: `28c2d16f08ccbfb0d0279003eace5431b5401619`
Runner SHA-256: `c26e284cab6fd16ac2e8c971df64211c13fe14d4e935a5e023470d311a650e77`
GitHub Actions run: `31575673964`, job `94047117499`
Raw artifact ID: `9133099180`
Raw JSON SHA-256: `71e0fa1f9ccfd0f772c1399f0e5c53e2843d6cd084ca8e357a568e0ba6f173f4`
Raw log SHA-256: `a8dd64d1bd7abfa4630cede71b828d99dd55a65c45dfcff6529dea349281b130`

## Qualification prerequisite

Czech had legitimately unlocked target scoring by passing all three frozen positive-control scales before H19 access:

| K | Czech control rank | margin | map accuracy | paired agreement | converged | oracle gap |
|---|---:|---:|---:|---:|---|---:|
| 22 | 1 | 0.1520872407 | 1.0 | 1.0 | yes | 0.0 |
| 26 | 1 | 0.1477822467 | 1.0 | 1.0 | yes | 0.0 |
| 36 | 1 | 0.1521850012 | 1.0 | 1.0 | yes | 0.0 |

Thus the negative H19 result below is not attributed to an unqualified Czech inference instrument.

## Binding H19 family result

Czech fit to the exact frozen RF STA-family K=22 H19 target:

- Czech held-out score: **-2.7238400484645537** nats/retained unit;
- Czech rank after insertion into the immutable v1.9 comparator: **8 of 9**;
- Czech paired-map agreement: **1.0**;
- Czech convergence: **PASS** after the first six-restart ensemble batch;
- H19 coverage: **0.9983818770226537**;
- H19 word coverage: **0.9980126692336355**.

Nine-language ranking:

1. Spanish: -2.5711088593
2. French: -2.5934305936
3. Greek: -2.6259873716
4. German: -2.6319475217
5. Latin: -2.6527180797
6. Hebrew: -2.6835728413
7. Italian: -2.7021448099
8. **Czech: -2.7238400484645537**
9. Arabic: -2.7801834577

The top-vs-runner margin remains the original Spanish-vs-French value:

**0.0223217343 nats/unit < frozen 0.05 threshold.**

Therefore H19 family **FAILS** exactly as v1.9 did, and Czech does not alter the prior scientific verdict.

## Stopping rule

Per the binding pre-target protocol:

- connected-AAA Czech target fit: **NOT OPENED**;
- full-STA Czech target fit: **NOT OPENED**;
- C19/plaintext: **SEALED / NOT OPENED**.

No post-hoc Czech-specific normalization, key, representation, corpus subset, optimiser setting or language threshold is permitted in this diagnostic.

## Interpretation

Czech is a useful negative control rather than a new language lead. The M19/STA inference instrument can recover Czech perfectly on synthetic positive controls at K=22/26/36, but the actual Voynich H19 target does not preferentially select Czech. Czech scores below seven of the original eight candidate languages and cannot repair the original family-level margin failure.

This diagnostic therefore leaves the prior cipher closeout unchanged: **NO STA/AAA M19 SIGNAL**.

## Branch-hygiene note

The initial Czech protocol file was accidentally written onto the closed cipher-closeout branch before any Czech target scoring. A dedicated branch `experiment/m19-czech-diagnostic-v1-20260812` was created from that protocol commit, and the original closeout branch was force-restored to its exact prior head `418da5635ffa2b1e86053dfd49fc1022ba15c297`. Thus the closed branch's content/state remains unchanged; the add-and-restore history is retained in repository objects for auditability.
