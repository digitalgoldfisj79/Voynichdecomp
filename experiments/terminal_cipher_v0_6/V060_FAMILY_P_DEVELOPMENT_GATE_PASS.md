# v0.6 Family P — corrected full development result

Date: 2026-07-16

Verdict: **DEVELOPMENT GATE PASSED. LOCKED TEST AUTHORISED.**

No locked-test result or Voynich data was inspected before this development decision.

## Execution provenance

- Corrected full-development job: `Digitalgoldfish79/6a59350a85d9643ce16d6a94`
- Job Git head: `f806d6974482fc99676a129b289f2beae0bef624`
- Runtime: 504 seconds
- Execution clarification: `b4f7f556766e29e8aeadf85b3f596e7f81bfac7b`
- Scientific result SHA-256: `4fd8b789e40c179094c1e414a15e602eac0a5ea4bd72afced0197d032eb4357a`

The only correction was finite handling of the impossible period-2 uniqueness request. The frozen eight-start budget, modes, periods, search objective, iteration counts, restart counts, penalties, data and gates were unchanged.

## Development results

| Metric | Result | Required |
|---|---:|---:|
| Trials | 16 | 16 |
| Mean plaintext recovery | **99.5117%** | ≥80% |
| Median plaintext recovery | **99.6094%** | ≥90% |
| Minimum plaintext recovery | **98.1771%** | — |
| Trials ≥80% | **16/16** | ≥14/16 |
| Trials ≥90% | **16/16** | — |
| Exact plaintexts | **4/16** | — |
| Mode accuracy | **16/16** | ≥14/16 |
| Period accuracy | **16/16** | — |
| Full structure accuracy | **16/16** | ≥12/16 |

All registered development conditions passed with substantial margin.

## Consequence

Family P may proceed to its sealed locked-test set using the identical frozen solver and the already-recorded execution-only termination correction. No tuning or threshold modification is permitted between development and locked test.
