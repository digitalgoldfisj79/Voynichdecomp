# Fractionation Composition — Running Results

## Retractions / superseded findings

1. **RETRACTED AS A DISCRIMINATOR — phase periodicity alone.** The v0.1 detector produced very large z scores for planted fractionation, but the full six-language development run also sent 56.25% of negative controls above the preregistered z >= 3 threshold. In particular, 100% of verbose-monographic controls and 76.39% of verbose-monographic+transposition controls crossed the threshold. Therefore a large phase-MI residual is not specific evidence for coordinate fractionation. It measures periodic/verbose expansion structure more generally.

2. **SUPERSEDED / NOT PROMOTED — v0.1a coordinate-structure score.** Adding Cartesian pair reuse and matching control alphabet size substantially reduced false positives, but did not meet the preregistered specificity requirement. It must not be presented as a validated fractionation detector.

These retractions concern detector interpretation, not the implementation fact that planted coordinate fractionation has phase and pair structure.

## Audit correction: UD test access

The protocols intended the UD `test` split to remain sealed during development. The implementation did **not** satisfy that strict operational seal: the shared corpus loader downloaded, hash-checked, parsed and encoded `train`, `dev` and `test` on each run before the experiment selected `dev_words` for all evaluated samples.

No test sample entered either detector, no test statistic/gate was computed, and no test result was inspected or used for tuning. Therefore this is not evidence leakage into the reported development statistics, but it is an audit/sealing defect and the test split must not be described as literally unopened or untouched. Any future locked programme would need a split-scoped loader that cannot read test data during development.

Voynich data were not loaded or evaluated by this branch.

## v0.1 — preregistered phase-only synthetic gate

Status: **FAILED / STOP_NON_IDENTIFIABLE for this statistic**.

Full development run:
- six pinned UD development corpora: en, de, fi, tr, he, ar;
- 12 replicates per language;
- target 600 plaintext letters per sample;
- 99 matched positional permutation nulls per sample;
- positives: 100% at z >= 3;
- controls: 56.25% at z >= 3 (gate <=10%);
- positive mean z = 123.8076613;
- control mean z = 3.9461263;
- control z SD = 3.1929241;
- mean separation = 119.8615349 z units = 37.5397 control SDs.

The large mean separation does not rescue the statistic because the preregistered decision-rule false-positive rate is unacceptable.

Control threshold rates:
- slot_control: 33.33%;
- expanded_mono: 100%;
- expanded_transposition: 76.39%;
- markov_control: 15.28%.

Result SHA-256: `6f545356eb698fada3270b78c1430d33e33758176594aec4cf144462c93b90ab`.

## v0.1a — coordinate-structure development amendment

Status: **FAILED / STOP_NON_IDENTIFIABLE. BRANCH CLOSED BEFORE ANY UD-TEST EVALUATION OR VOYNICH RUN.**

Development parameters:
- same six pinned UD development corpora;
- 4 replicates per language;
- target 400 plaintext letters per sample;
- 39 matched positional permutation nulls per sample;
- control surface inventory matched to `rows + columns`;
- statistic: max over token/stream b=1..8 of `phase_MI * pair_density * (1 - pair_NMI)`.

Aggregate result:
- positives: 100% at z >= 3;
- controls: 19.7917% at z >= 3 (gate <=10%);
- positive mean z = 129.8475621;
- control mean z = 1.9127836;
- control z SD = 2.0883949;
- mean separation = 127.9347785 z units = 61.2599 control SDs.

Again, the very large mean separation does not rescue the detector because the preregistered control false-positive rate remains nearly twice the permitted ceiling.

Control threshold rates:
- slot_matched: 0%;
- expanded_shared: 37.5%;
- expanded_shared_trans: 29.1667%;
- markov_matched: 12.5%.

Positive threshold rates:
- frac_pair: 100%;
- frac_block_token: 100%;
- frac_block_stream: 100%;
- frac_homophonic: 100%.

Result SHA-256: `815e36dc53256f6e8fdb535ad28d57d13c726c96ccaa115fb9526244c0bfabf8`.

### Determinism check

A second CI execution on a different hosted runner/region reproduced the v0.1a output exactly, including result SHA-256 `815e36dc53256f6e8fdb535ad28d57d13c726c96ccaa115fb9526244c0bfabf8`. The failure is therefore deterministic under the pinned code/data/seeds rather than a runner fluctuation.

## Branch decision

The branch is closed at the synthetic-development stage. No UD-test evaluation was performed and Voynich was not run.

Reason: two successive development detectors were sensitive to planted fractionation but insufficiently specific against difficult verbose/shared-alphabet controls. Further tuning on the same development controls would materially increase circularity and decision-rule overfitting. A future revisit would require a genuinely new preregistered discriminator or independent theoretical invariant, not another threshold/statistic adjustment to these development results.

Current inference: the proposed fractionation+regrouping family remains **not excluded**, but this programme did **not** obtain a valid distinguishing test with which to ask whether Voynich is compatible with it. No positive Voynich evidence was generated.
