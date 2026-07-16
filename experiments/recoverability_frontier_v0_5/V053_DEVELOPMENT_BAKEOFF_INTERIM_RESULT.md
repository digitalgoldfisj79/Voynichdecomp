# Recoverability frontier v0.5.3 — development bake-off interim result

Date: 2026-07-16

Status: **development complete for four arms; million-example hybrid confirmation and locked tests active**

No Voynich text was scored.

## Frozen environment

- English bounded frequency-adaptive homophonic substitution;
- 384 normalized plaintext characters;
- corpus `dev` replicates 0–7;
- fresh key per example except the explicitly separate shared-pool positive control;
- first-occurrence recurrence canonicalisation;
- inferred homophone inventory, never the true inventory;
- train-only language models.

## Arm A — classical restart curve

Job: `Digitalgoldfish79/6a585bafb1669a49bf076b69`

Scientific SHA-256: `bdb967dfbfbacdfe62b4417c251026cc3f54a3d75ac3fb9808416ea869e16e12`

Strict fixed-inventory CrypTool-style trajectories, `3,000,000` pair proposals per restart:

| Restarts | Mean recovery | Median | ≥70% trials | ≥95% trials | Gate |
|---:|---:|---:|---:|---:|---|
| 12 | 55.5339% | 57.0313% | 4/8 | 4/8 | fail |
| 24 | 56.9661% | 57.6823% | 4/8 | 4/8 | fail |
| 48 | 67.5456% | 98.5677% | 5/8 | 5/8 | fail |
| 96 | 67.8711% | 98.5677% | 5/8 | 5/8 | fail |
| 192 | **89.6159%** | **99.3490%** | **7/8** | **7/8** | **pass** |

The minimum development-eligible classical schedule is 192 restarts.

## Arm B — neural-language-model beam search

### One-layer LSTM

Job: `Digitalgoldfish79/6a585bbab1669a49bf076b6d`

Scientific SHA-256: `5d95943663bff845fe48173bb7f62d42f9558b17951e5bc0d71b846169baa97a`

Best result: beam 2048, mean recovery **26.2695%**, median 21.8750%. No trial reached 70%.

### Two-layer LSTM

Job: `Digitalgoldfish79/6a585bc485d9643ce16d5cb4`

Scientific SHA-256: `beeaaaf5bef5a7a254c31368d05504a897eea667c9f059c34a39afcb31db02aa`

Best result: beam 2048, mean recovery **24.2188%**, median 20.9635%. No trial reached 70%.

Both language models achieved low train loss, but first-occurrence assignment beam search pruned the correct key path. The whole-sequence neural-LM beam arm is stopped.

## Arm C/E — fresh-key recurrence Transformer and hybrid

### Smaller diagnostic

Job: `Digitalgoldfish79/6a585bcdb1669a49bf076b6f`

Scientific SHA-256: `d7aaff10b663846ec332f748ca7769207397fa283cd9f326dc1cb53b670bf937`

- direct greedy recovery: 22.0378%;
- inventory-constrained recovery: 27.2135%;
- neural-seeded classical hybrid: 68.7826% mean, 98.5677% median, 5/8 above 95%;
- reliability gate: fail.

### Larger diagnostic

Job: `Digitalgoldfish79/6a585bd785d9643ce16d5cb6`

Scientific SHA-256: `cfe953b5a57149de601ddba0eba218ecfb0f446721e6030f9e4a27b18fac9c54`

- direct greedy recovery: 19.1732%;
- inventory-constrained recovery: 28.4831%;
- neural-seeded classical hybrid: **99.3164% mean**;
- median: **99.4792%**;
- all 8/8 trials above 95%;
- development gate: **pass**.

The neural model is not an adequate standalone decipherer. Its posterior structure is nevertheless an effective basin-selecting prior. Thirty-two posterior-derived keys followed by only `250,000` classical proposals per seed outperform 192 blind classical restarts decisively in both reliability and search cost.

The selected architecture is being repeated with `22,000 × 48 = 1,056,000` freshly generated training examples, as required by the frozen protocol.

## Arm D — shared-pool positive control

Job: `Digitalgoldfish79/6a585be1b1669a49bf076b73`

Scientific SHA-256: `ac1380e8461431c9c04427405526fc44e0076374f61811355fedc31525ead1d4`

- mean recovery: **100%**;
- exact recovery: **20/20**;
- positive-control gate: pass.

This verifies the easier reused-code-pool regime but is ineligible as fresh-key evidence. The gap between 100% shared-pool recovery and weak direct fresh-key recovery is scientifically substantive.

## Development decision

Eligible fresh-key arms:

1. classical 192-restart solver;
2. large recurrence-Transformer posterior plus classical hybrid.

Ineligible/stopped:

- one-layer neural-LM beam;
- two-layer neural-LM beam;
- standalone recurrence Transformer;
- shared-pool LSTM as fresh-key evidence.

## Active next steps

- Classical 192-restart locked test is running on untouched English test replicates 128–147.
- Million-example hybrid confirmation is running.
- If the million-example confirmation preserves the development gate, the identical hybrid configuration will be run once on the same untouched locked test block.
- The locked gate remains: mean ≥70%, median ≥90%, and at least 16/20 trials ≥70%.
