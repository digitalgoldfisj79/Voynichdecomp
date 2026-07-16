# v0.6 Family P — locked-test result

Date: 2026-07-16

Verdict: **LOCKED TEST PASSED. VOYNICH APPLICATION AUTHORISED.**

No Family P tuning, threshold change or scientific amendment was made between development and locked test. Voynich data was not inspected before this decision.

## Provenance

- Development gate report commit: `0dd4a5121e9e8070037ccd5760c821f123a58f12`
- Development result SHA-256: `4fd8b789e40c179094c1e414a15e602eac0a5ea4bd72afced0197d032eb4357a`
- Locked-test job: `Digitalgoldfish79/6a59378bb1669a49bf078374`
- Locked-test Git head: `6e968c190a2407eb9edcd56d746d4f6e56e3b87f`
- Locked-test runtime: `539` seconds
- Locked-test result SHA-256: `f16e35a7c058a44e7f864189d6da6172292e0b42ae8a0d200263eacdbbeb3ed5`
- Seed-termination execution clarification: `b4f7f556766e29e8aeadf85b3f596e7f81bfac7b`

The execution clarification preserved eight stochastic starts. It corrected only the impossible requirement for eight unique vectors when candidate period 2 has five possible histogram-start vectors.

## Locked-test results

| Metric | Result | Registered requirement |
|---|---:|---:|
| Trials | 16 | 16 |
| Mean plaintext recovery | **94.3197%** | ≥80% |
| Median plaintext recovery | **100.0000%** | ≥90% |
| Minimum plaintext recovery | **10.6771%** | — |
| Trials ≥80% | **15/16** | ≥14/16 |
| Trials ≥90% | **15/16** | — |
| Exact plaintexts | **13/16** | — |
| Mode accuracy | **16/16** | ≥14/16 |
| Period accuracy | **15/16** | — |
| Full structure accuracy | **15/16** | ≥12/16 |

All registered locked-test conditions passed.

## Mandatory outlier disclosure

One trial failed catastrophically:

- split: `test`;
- true mode: `periodic`;
- replicate: `0`;
- true period: `10`;
- selected mode: `periodic`;
- selected period: `2`;
- mode correct: `true`;
- period and full structure correct: `false`;
- final plaintext recovery: **10.6771%**;
- screening-best recovery: **7.5521%**.

The remaining 15 trials recovered between **99.2188% and 100%**. No post-test repair, rerun, period-specific exception or threshold modification is permitted. The outlier remains part of the reported locked-test distribution.

## Interpretation

Family P is demonstrably recoverable under fresh hidden base alphabets, unseen schedules, blind periodic-versus-line-reset mode selection and unseen locked-test periods. Its failure mode is sharply bimodal rather than smoothly degraded: fifteen trials were essentially solved, while one period-10 trial converged on an incorrect period-2 structure and failed.

This means a high-scoring Family P fit to Voynich can be scientifically interpreted only together with structural stability, competing-period margins, restart agreement and section/line robustness. A single best-scoring decode is not sufficient evidence.

## Protocol consequence

Family P is the sole terminal v0.6 family that passed both development and locked test. It is therefore authorised for the frozen Voynich application. Families T, G and S remain closed and may not be applied.
