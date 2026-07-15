# Recoverability frontier v0.5.0 — full learned-decoder result

Date: 2026-07-15

Verdict: **FAIL; DO NOT EXPAND THIS ARCHITECTURE**

No Voynich text was evaluated.

## Frozen execution

- Branch: `experiment/recoverability-frontier-v0.5.0-20260715`
- Frozen protocol: `RECOVERABILITY_PROTOCOL_V050.md`
- Protocol amendment: `PROTOCOL_AMENDMENT_V050_A.md`
- Optimized recurrence launcher commit: `54257e91338ca0ca9ec6350c802b0ba38d3dab28`
- Family-known job: `Digitalgoldfish79/6a57f1a885d9643ce16d5648`
- Blind-family job: `Digitalgoldfish79/6a57f1b085d9643ce16d564a`
- Hardware: one `a100-large` per job, run in parallel

The channel-oracle gate had already passed all eight noiseless cipher families at 100% mean character recovery.

## Training behaviour

### Family-known

| Epoch | Total loss | Sequence loss | Classification loss |
|---:|---:|---:|---:|
| 1 | 3.3006 | 2.5939 | 0.7067 |
| 2 | 2.9186 | 2.2257 | 0.6929 |
| 3 | 2.6241 | 1.9369 | 0.6872 |
| 4 | 2.4515 | 1.7681 | 0.6834 |

### Blind-family

| Epoch | Total loss | Sequence loss | Classification loss |
|---:|---:|---:|---:|
| 1 | 3.2982 | 2.5912 | 0.7070 |
| 2 | 2.9236 | 2.2292 | 0.6944 |
| 3 | 2.6371 | 1.9448 | 0.6923 |
| 4 | 2.4661 | 1.7749 | 0.6912 |

The sequence objective learned, but the message classifier remained close to the binary cross-entropy value of an uninformative balanced classifier (`ln 2 ≈ 0.6931`).

## Development-frozen thresholds

| Arm | Threshold | Development sensitivity | Development FPR |
|---|---:|---:|---:|
| Family-known | 0.573001 | 8.125% | 5.000% |
| Blind-family | 0.546866 | 5.289% | 4.988% |

## Held-out test results

| Arm | Positive trials | Control trials | Sensitivity | Control FPR | Mean accuracy, all positives | Mean accuracy, detected positives | Exact recovery |
|---|---:|---:|---:|---:|---:|---:|---:|
| Family-known | 8,640 | 8,640 | 9.028% | 5.162% | 2.161% | 23.939% | 0% |
| Blind-family | 8,640 | 8,640 | 7.014% | 4.861% | 1.673% | 23.854% | 0% |

### Family-known by cipher family

| Family | Sensitivity | Mean accuracy over all positives | Exact recovery |
|---|---:|---:|---:|
| `mono` | 15.926% | 3.853% | 0% |
| `homophonic` | 0.556% | 0.132% | 0% |
| `null_homophonic` | 0.926% | 0.236% | 0% |
| `polyalphabetic` | 0.000% | 0.000% | 0% |
| `feedback` | 0.093% | 0.024% | 0% |
| `nomenclator` | 42.222% | 9.985% | 0% |
| `transposition` | 10.741% | 2.633% | 0% |
| `fractionated` | 1.759% | 0.426% | 0% |

### Blind-family by cipher family

| Family | Sensitivity | Mean accuracy over all positives | Exact recovery |
|---|---:|---:|---:|
| `mono` | 8.426% | 1.972% | 0% |
| `homophonic` | 5.278% | 1.266% | 0% |
| `null_homophonic` | 5.278% | 1.275% | 0% |
| `polyalphabetic` | 0.556% | 0.130% | 0% |
| `feedback` | 1.481% | 0.351% | 0% |
| `nomenclator` | 21.852% | 5.273% | 0% |
| `transposition` | 7.593% | 1.787% | 0% |
| `fractionated` | 5.648% | 1.330% | 0% |

## Frozen gate decision

The family-known arm failed all three required gates:

- sensitivity was below 80%;
- test FPR exceeded 5%;
- fewer than five families reached 70% mean character accuracy.

The blind-family arm met only the test FPR requirement and failed sensitivity and recovery requirements.

Therefore v0.5.0 does not justify broader language expansion or application to real unknown text.

## Critical methodological diagnosis

The failed message classifier was trained to distinguish:

1. held-out corpus text transformed by a cipher; from
2. a generated latent sequence transformed by the same cipher.

Both are legitimate plaintext sequences at the cryptographic channel level. Whether a sequence was “independently selected” or emitted by a generator is not encoded in the ciphertext. The classifier loss remaining near `ln 2` is therefore consistent with the non-identifiability already established by v0.3–v0.4.

This classification failure must not be interpreted as evidence that plaintext recovery is impossible. It shows that the v0.5.0 abstention target was conceptually invalid.

The generic Transformer also performed poorly as a decoder on unseen independent keys. That is a separate model-capacity and solver-mismatch result. One generic architecture cannot define the recoverability frontier.

## Required pivot

v0.5.1 must:

- remove the generated-versus-selected message classifier from the primary recovery task;
- treat every enciphered latent sequence as having a recoverable ground-truth plaintext, including generated controls;
- evaluate exact and approximate recovery directly;
- calibrate abstention from predicted recovery reliability, not alleged messagehood;
- use a development-frozen portfolio of family-specialist cryptanalytic solvers;
- report `UNSUPPORTED_FAMILY` or `LOW_RECOVERY_CONFIDENCE` rather than `NO_MESSAGE` when the channel cannot be solved;
- retain a separate bounded model-comparison programme for cipher-family versus generator-family explanations.

## Provenance limitation

The complete row-level JSON files were written only to ephemeral job filesystems and were not uploaded. The aggregate training, calibration, family-level and gate results are preserved in immutable Hugging Face logs and in this report. Future runs must use `run_and_emit_v050.py` or a writable bucket to preserve compressed row-level artifacts through job logs.
