# v0.3.4 generator-disjoint smoke result

Date: 2026-07-15

Verdict: **FAIL — STOP BEFORE LARGE LOCKED VALIDATION**

## Frozen protocol

Protocol commit: `65bac67c5ec8df878f7377959a10243f6e32aa45`

Scientific implementation commit: `41009d8577a069339b573b84018ad1087d7285b2`

Hugging Face job: `Digitalgoldfish79/6a57dcb285d9643ce16d547c`

Primary result JSON SHA-256 emitted by the job:

`5e427f045ff76d9dd6871299b1cd5247765341860b1ad83d94b86ac7d4f94770`

The post-run temporary-host upload failed because `curl` was not installed in the container. The complete primary outcomes were recovered from the immutable job log and are preserved in `v034_generator_disjoint_smoke_primary.csv`. The failed upload occurred after the summary and SHA had been emitted and did not affect computation.

## Design

The fixed v0.3.3 beam detector and 199-permutation latent-order test were applied without threshold changes.

Positives were independently generated homophonic encipherments of ordered words from two Greek corpora that were not used to fit the detector's Currier-I external model. Four independently coded surface mechanisms were used: keyed PRF, rotor, feedback with nulls, and line-keyed homophony with nulls.

Controls used the same 24-cell surface interface but contained no plaintext message. Their latent sequences came from an ordered HMM, a motif grammar, a topic finite-state process, or copied/mutated latent lines.

## Primary results

| Class | Detected | Trials | Rate |
|---|---:|---:|---:|
| Independent enciphered Greek positives | 2 | 16 | **12.5% sensitivity** |
| Ordered non-message controls | 10 | 16 | **62.5% false-positive rate** |

Specificity was **37.5%**.

95% Wilson intervals:

- sensitivity: 3.5%–36.0%;
- false-positive rate: 38.6%–81.5%.

## Positive mechanisms

| Mechanism | Detected | Trials |
|---|---:|---:|
| keyed PRF, global | 0 | 4 |
| feedback, global, null-bearing | 0 | 4 |
| line-keyed, Currier-split, null-bearing | 0 | 4 |
| rotor, Currier-split | 2 | 4 |

Both Greek corpora were detected at only 1/8.

## Ordered controls

| Control family | False positives | Trials |
|---|---:|---:|
| motif grammar | 4 | 4 |
| copy/mutate latent lines | 3 | 4 |
| topic finite-state process | 2 | 4 |
| ordered HMM | 1 | 4 |

## Interpretation

The v0.3.3 statistic is not a general detector of encoded language. It measures compatibility between a recovered partition and the fixed Currier-I-derived external transition model.

The earlier 73/96 sensitivity and 6/64 false-positive result did not generalize because its positives were generated from the same broad latent transition family used by the detector, while its controls mostly lacked deliberately ordered latent dynamics.

When confronted with independently generated enciphered language, v0.3.3 rejected 14/16 cases. When confronted with ordered hidden-state processes carrying no plaintext, it accepted 10/16 cases. This is the reverse of the required behaviour.

The result rules out a large locked validation of the present comparator. More beam, heuristic, parallel-tempering or neural compute cannot repair this conceptual failure.

## Scientific boundary

The current programme can detect some forms of latent sequential order. It cannot distinguish an encoded source message from structured non-message generation, and it is highly dependent on the assumed external transition model.

Therefore:

- do not run v0.3.3 or v0.3.4 on the Voynich Manuscript;
- do not scale the present detector;
- do not interpret a positive latent-order result as evidence for cipher;
- redesign the target as a source-message-transfer or independent-language predictability test rather than a fixed external-model order test.
