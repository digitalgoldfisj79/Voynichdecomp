# SELECT -> FORM normalized calibration class K4
Date: 2026-10-04
Status: FROZEN AFTER K4a DISCOVERY SCREEN
Scope: synthetic recoverability only. NO P70. No real Voynich inversion.

## Parent results
- voynich_select_form_phaseK3c_panel_result_20261004
- K4a HF job 6ac2a4c8fbc85ba6823a1932
- K4a script commit 2c43acf5ff8b619b4e8eb50ae98dc27beb2f43d2

## Why normalization is required
The unnormalized K3c 20-seed panel mixed two failure modes:
1. some planted instances were intrinsically weak: oracle NMI fell as low as ~.58;
2. on informative instances, blind search still sometimes missed recoverable structure.

The calibration ensemble therefore needs a fixed, truth-independent difficulty definition before blind inversion.

## Frozen socket
K=8 source states.
Sparse source graph out-degree d=4.
Rank-2 SELECT control.
ENTRY strength=5.0.
ROUTE strength=.5.
ROUTE biases only frozen legal FORM destination classes.
FORM legality, STOP and exact-piece realization remain frozen.

## Discovery screen K4a
96 independent planted parameter draws.
Median oracle NMI .7422.
10th percentile oracle NMI .6390.
68/96 oracle NMI >= .70.

Strongest simple parameter-space predictor:
q10_pair_js = 10th percentile of pairwise state divergence under the SELECT socket
(entry JS + real-FORM-exposure-weighted ROUTE JS).
Correlation with oracle NMI ~.386.

## Frozen normalized acceptance rule
A planted parameter draw is accepted iff, BEFORE blind search:

1. q10_pair_js >= 0.30;
2. stationary_min >= 0.015;
3. synthetic mean token length / real FORM mean token length is in [0.80, 1.20].

The acceptance rule DOES NOT use:
- planted hidden labels;
- oracle decoded labels;
- oracle NMI/ARI;
- any blind-search result.

Mean token length is observable surface behavior and is used only to prevent ROUTE from creating an artificial length code.

stationary_min is computed from the planted source transition matrix before data generation and ensures all source states carry material probability mass.

## Discovery-set audit
On K4a, 7/96 draws met the frozen normalized rule.
All 7 had oracle NMI >= .792.
This is discovery evidence only and must not be counted as confirmation.

## Confirmation sequence
K4b:
- scan fresh parameter seeds;
- accept the first 20 draws satisfying the frozen rule;
- do not use oracle metrics to accept/reject;
- after the 20 accepted instances are frozen, reveal oracle metrics only for calibration audit.

K4b oracle gate:
- median oracle NMI >= .75;
- 10th percentile oracle NMI >= .70;
- >=90% of accepted instances oracle NMI >= .70.
If this fails, do not run blind K4c and do not retune on K4b.

K4c:
- run identical frozen K3b blind pipeline on the 20 K4b accepted instances;
- no instance-specific tuning;
- refit is no longer automatically privileged: pre-refit and post-refit are both reported, but final choice rule must be frozen prospectively before K4c evaluation.

Recovery-contract gates remain:
- median heldout NMI >= .70;
- 10th percentile >= .60;
- >=90% replicates NMI >= .60;
- median transition-edge F1 >= .70;
- adequate validation-selected solution stability.

## Interpretation firewall
Passing K4b/K4c would validate only the recoverability of this SELECT -> FORM socket under a normalized informative synthetic ensemble.
It would not establish semantics or identify the historical upstream content.
