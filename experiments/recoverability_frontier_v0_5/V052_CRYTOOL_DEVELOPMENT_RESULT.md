# Recoverability frontier v0.5.2 — independent CrypTool-style development result

Date: 2026-07-15

Verdict: **PASS ENGLISH 384-CHARACTER DEVELOPMENT; FREEZE FOR UNTOUCHED TEST**

No Voynich text was scored.

## Independent search architecture

The solver is a benchmark-specific Python port of the Apache-2.0 CrypTool 2 `HomophonicSubstitutionAnalyzer` search architecture, pinned at commit:

`d7d754af55c167941bec7fb56e965f309d050a12`

Relevant upstream sources:

- `HillClimber.cs`;
- `SimulatedAnnealing.cs`;
- `HomophonicSubstitutionAnalyzerSettings.cs`.

The strict benchmark adaptation uses:

- exhaustive pair sweeps;
- linear cooling;
- calibrated initial temperature;
- acceptance-probability floor `0.0085`;
- independent restarts;
- exact preservation of the inferred homophone-label multiset in every restart;
- no rare-symbol inventory mutation;
- incremental train-only quadgram scoring.

## Development grid

English, 384 normalized characters, 8 development chunks and unseen keys.

| Steps × restarts | Target acceptance | Mean recovery | Median | Inventory overlap | SHA-256 |
|---|---:|---:|---:|---:|---|
| 1,000,000 × 8 | 0.05 | 57.5195% | 59.8958% | 95.8729% | `1c35f5807946a6edefea08e02da3e7bbbd242c1278c42bad24d5ca346f0ab404` |
| 1,000,000 × 8 | 0.20 | 24.6745% | 12.7604% | 95.8729% | `3d3865ec16686aefa05a6d59061acd47ae0bab42516f4cc92aca781f09edba6a` |
| 3,000,000 × 12 | 0.05 | **78.5807%** | **99.3490%** | 95.8729% | `3292e2eeb4564bb6c1ab423fa85b75d16627d53503a61f32458be71f5cb15713` |

Selected development job:

`Digitalgoldfish79/6a58130585d9643ce16d598d`

The low-temperature result clears the frozen 70% English development gate. The high-temperature schedules are rejected.

## Interpretation

The English long-text failure was not intrinsic to the language model or homophonic family. It resulted from the proposal schedule and inventory instability of the project-local solvers.

The successful combination is:

1. preserve the already strong inferred inventory;
2. perform exhaustive rather than random pair sweeps;
3. use many complete independent trajectories;
4. cool conservatively from a scale-calibrated low initial acceptance rate.

## Frozen test

The selected schedule is fixed at:

- 3,000,000 accepted-or-rejected pair proposals per restart;
- 12 restarts;
- target initial acceptance 0.05;
- no inventory mutation.

It will be applied once to 20 untouched English test chunks of length 384 beginning at replicate offset 96. No test-time tuning is permitted.
