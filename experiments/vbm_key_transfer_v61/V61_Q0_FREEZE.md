# VBM v6.1 — Q0/Q2 freeze before Voynich FIT

Date: 2026-08-11
Namespace: `VBMKEYTRANSFERV61`
HF formal CAL/VAL job: `6a7b936f27caad61c6eac6da`

## Binding qualification result

Synthetic CAL separated prospectively and froze:

- `TAU_ITE = 0.7364209334512819`
- `TAU_EKS = 0.6750480875047778`

CAL extrema:
- weakest reusable-key positive ITE = 0.8577435465496404
- strongest negative/adversarial ITE = 0.6150983203529234
- weakest reusable-key positive EKS = 0.7571863910746477
- strongest fresh-key EKS = 0.592909783934908

Untouched VAL:
- reusable-key positives: **12/12 pass both thresholds**
  - BAV_GLOBAL 4/4
  - GER_GLOBAL 4/4
  - BAV_GLOBAL_SWAP 4/4
- negative/adversarial false positives: **0/12**
  - BAV_FRESH 0/4
  - GER_FRESH 0/4
  - STABLE_MARKOV 0/4

Therefore v6.1 is qualified for the preregistered Voynich FIT stage.

## Target discipline

- H1 is not reused.
- C1 remains sealed.
- FIT may now be scored under six deterministic held-out folio folds.
- No target plaintext or per-symbol argmax mapping may be inspected.
- FIT pass requires all preregistered aggregate/fold gates from `VBM_KEY_TRANSFER_V61_PROTOCOL.md`.
- C1 may be opened only if FIT passes.
