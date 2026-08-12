# M19 Czech Diagnostic v1.1 — binding pre-target protocol

Date: 2026-08-12
Supersedes `PROTOCOL.md` **before any Czech Voynich target scoring**.
Reason for revision: avoid unnecessary upstream-corpus drift in the already-frozen eight-language comparator.

All mechanism, normalization, optimizer, qualification, representation, margin, convergence and stopping rules from v1 remain unchanged except the target-comparator implementation below.

## Frozen prior family comparator

The original v1.9 RF H19 STA-family result is treated as immutable historical data:

1. Spanish: -2.5711088593
2. French: -2.5934305936
3. Greek: -2.6259873716
4. German: -2.6319475217
5. Latin: -2.6527180797
6. Hebrew: -2.6835728413
7. Italian: -2.7021448099
8. Arabic: -2.7801834577

Original Spanish-vs-French margin: 0.02232173435 nats/retained unit.
Original H19 coverage: 0.9983818770.
Original family fits: all converged; original top-language paired-map agreement = 1.0.

Source: `experiments/bnf_m19_sta_hierarchy_v1_9/H19_RESULT.md`, frozen before this Czech experiment.

## Revised Czech target comparison

If Czech passes fresh positive-control qualification at K=22, K=26 and K=36:

1. Reconstruct the exact v1.9 RF H19 STA-family T/H split and K=22 vocabulary using the pinned v1.9/v1.7 code and frozen Voynich source hashes.
2. Build the Czech LM from pinned `UD_Czech-CAC` commit `798f89716ae5a96e86042df7d394d56787e2e213`, using the unchanged M19 normalization and LM-construction rule.
3. Fit **Czech only** to the v1.9 family training arm with the exact v1.9 paired optimizer and convergence rule.
4. Score Czech on the untouched H19 arm.
5. Insert that Czech score into the immutable original eight-language ranking above.
6. Compute the nine-language top and top-vs-runner margin.

This is preferable to rerunning the original eight because their published/frozen scores remain exactly unchanged and cannot be affected by upstream language-corpus changes.

Family gate remains exactly:
- H19 coverage >= 0.97;
- nine-language top-vs-runner margin >= 0.05;
- if Czech is top: Czech paired-map agreement >=0.90 and Czech convergence PASS;
- if an original language remains top: its frozen v1.9 convergence/agreement conditions remain PASS and Czech must additionally converge;
- all nine fits are considered converged only if the eight frozen fits passed (they did) and the new Czech fit converges.

### Consequence

If Czech scores at or below the frozen Spanish score, the pre-existing Spanish-French margin remains 0.02232173435 < 0.05 and family H19 necessarily fails. Therefore no AAA/full-STA target runs are allowed.

If Czech scores above Spanish, family H19 passes only if Czech exceeds Spanish by >=0.05. A smaller Czech lead remains a failed margin gate.

Only if family passes may a new preregistered AAA target extension be executed, followed sequentially by full STA. C19 remains sealed.

## Qualification remains unchanged

Czech must first pass fresh controls at all three frozen K values (22/26/36), ranking Czech against the nine language LMs on Czech held-out control text with:
- Czech rank 1;
- margin >=0.05;
- mapping accuracy >=0.85;
- paired agreement >=0.90;
- convergence PASS;
- best-minus-true oracle score >= -1e-6.

No Voynich Czech target fit is permitted until all three pass.
