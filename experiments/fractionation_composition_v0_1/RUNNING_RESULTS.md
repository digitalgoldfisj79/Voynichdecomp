# Fractionation Composition — Running Results

## Retractions / superseded findings

1. **RETRACTED AS A DISCRIMINATOR — phase periodicity alone.** The v0.1 detector produced very large z scores for planted fractionation, but the full six-language development run also sent 56.25% of negative controls above the preregistered z >= 3 threshold. In particular, 100% of verbose-monographic controls and 76.39% of verbose-monographic+transposition controls crossed the threshold. Therefore a large phase-MI residual is not specific evidence for coordinate fractionation. It measures periodic/verbose expansion structure more generally.

This retraction concerns the detector interpretation, not the implementation fact that planted coordinate fractionation has phase structure.

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

Status: running on UD development split. UD test split and Voynich remain sealed.

Amendments made after, and because of, the v0.1 development failure:
- match all negative-control surface alphabets to the coordinate channel size (`rows + columns`);
- retain difficult verbose/shared-alphabet and transposed verbose controls;
- replace phase-only score with max over token/stream b=1..8 of:
  `phase_MI * pair_density * (1 - pair_NMI)`;
- retain the same positional matched-null framework and the same development gate.

Interpretation ceiling remains unchanged: even a later synthetic pass and Voynich positive would show structural compatibility only, not decryption or historical identification.
