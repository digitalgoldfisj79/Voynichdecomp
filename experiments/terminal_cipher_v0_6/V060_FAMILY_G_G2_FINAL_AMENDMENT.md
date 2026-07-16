# v0.6 Family G — final permitted G2 development amendment

Date frozen: 2026-07-16

Status: **FROZEN BEFORE AMENDED G2 RESULT**

## Trigger

The initial frozen G2 run, job `6a58d530b1669a49bf077861`, failed with:

- AUROC 0.675903;
- 32/64 payload covers detected;
- 50.0% mean recovery with abstentions scored zero;
- 32/64 recovered at 70% or better.

Every detected payload was exact in carrier class, exact in parameters and recovered at 100%. The detected set was the 32 plaintext payload covers; the 32 fresh-mono covers were systematically suppressed or misranked.

Initial result SHA-256:

`6d658fe149fde328319aec5a06309c51cfdb75e2c34afd34bb40bbe2728ed76e`

## Concrete defect

The initial encrypted arm was penalised twice:

1. the candidate mono score had a full key-description penalty subtracted before conversion to language-score evidence;
2. the resulting family maximum was then calibrated again against 256 matched null maxima, which already incorporate fresh-mono overfitting and the 2,935-rule search burden.

This placed correctly solved encrypted payloads on a systematically lower evidence scale than plaintext payloads and null maxima. In addition, only twelve substitution-invariant candidates entered mono refinement, causing some true encrypted carriers to be omitted before decipherment.

This is a score-scale and shortlist-capacity defect. It does not justify changing the carrier family, inventory, data, threshold construction or gates.

## Sole amendment

The final amended G2 solver makes exactly the following changes:

1. **Invariant shortlist expansion:** retain the four highest identity-language candidates and expand the substitution-invariant shortlist from 12 to 128 candidates.
2. **Mono screen budget:** increase shortlisted fresh-mono screening from 50,000 iterations × 5 restarts to 100,000 iterations × 8 restarts.
3. **Arm scale correction:** rank and detect the mono arm using its unpenalised train-language-model z-score. The key-description term remains computed and recorded as an MDL diagnostic and deterministic tie-breaker, but is not subtracted before the matched-null calibration.
4. **Invariant auxiliary weight:** reduce the invariant auxiliary contribution from 0.15 to 0.05 so it screens and breaks near-ties without imposing a second large negative scale shift on successfully deciphered mono candidates.
5. **Encrypted execution smoke:** the cheap preflight must run one deterministic monoencrypted payload and one matched null through the complete 2,935-rule inventory before the amended full development run.

The final 700,000-iteration × 50-restart recovery refinement is unchanged.

## Unchanged constraints

The following remain frozen:

- 2,935-candidate carrier inventory and its SHA-256;
- four cover generators;
- 64 payload covers and 256 matched null covers;
- payload length 96;
- fresh code/key generation;
- train/dev/test separation;
- third-highest-null threshold and strict `>` detection rule;
- all G2 gates;
- abstention treatment;
- no test or Voynich access.

This is the one and only permitted G2 development amendment. If the amended G2 run fails any frozen gate, Family G closes at development with no locked test and no Voynich application.
