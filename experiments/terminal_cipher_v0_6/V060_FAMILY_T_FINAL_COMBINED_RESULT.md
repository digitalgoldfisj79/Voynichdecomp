# v0.6 Family T — final combined development result

Date: 2026-07-16

Status: **FAILED AT DEVELOPMENT — FAMILY CLOSED**

## Provenance

- Repository: `digitalgoldfisj79/Voynichdecomp`
- Branch: `experiment/terminal-cipher-programme-v0.6-20260716`
- Protocol commit: `b751675e0ffdfa132579feacfe2f0d65f4884479`
- Final amendment commit: `2727018fa522f0730508655c7a56b5d09ac8f2b9`
- Final solver commit: `ebcbbde63ef7e148bc1bc936b89e95ba859e0d3e`
- Machine-readable combined result: `v060_family_t_final_combined_result.json`
- Combined scientific payload SHA-256: `dd388aff90108570413a99f60f99d3cb5ef5cd580ebdb7bfd16141b15b37311b`

Source Hugging Face shards:

| Job ID | Shard-reported SHA-256 |
|---|---|
| `6a58a1b6b1669a49bf077181` | `8ac93b63f9daf696dc0b0418aa4fc26923cc826eed5434248d867d634a921ea7` |
| `6a58a1c385d9643ce16d5fc1` | `5bd4a33d732224ea106156983ff2b5c617a34836e517a616cb6923ce79a30385` |
| `6a58a1d5b1669a49bf077183` | `19cfb5c3267cbec56f9ebc8c2b90b6ba571e742bdd680040d90713c199749ba5` |
| `6a58a1ebb1669a49bf077185` | `64ebe92b14705a6d388bf0d16b7586d1a80c554b47e6071b13367f75327e7d7f` |

## All 16 deterministic development trials

| Replicate | True mode | True width | Selected mode | Selected width | Recovery | Mode correct | Width correct | Permutation correct | Exact | Elapsed s | Source job |
|---:|---|---:|---|---:|---:|:---:|:---:|:---:|:---:|---:|---|
| 0 | global | 8 | global | 8 | 21.3542% | yes | yes | no | no | 858.370510 | `6a58a1b6b1669a49bf077181` |
| 0 | line_reset | 7 | line_reset | 10 | 20.8333% | yes | no | no | no | 863.970304 | `6a58a1b6b1669a49bf077181` |
| 1 | global | 7 | global | 7 | 24.4792% | yes | yes | no | no | 864.522493 | `6a58a1b6b1669a49bf077181` |
| 1 | line_reset | 8 | line_reset | 10 | 19.7917% | yes | no | no | no | 866.595207 | `6a58a1b6b1669a49bf077181` |
| 2 | global | 6 | global | 9 | 20.3125% | yes | no | no | no | 850.956031 | `6a58a1c385d9643ce16d5fc1` |
| 2 | line_reset | 8 | line_reset | 8 | 99.4792% | yes | yes | yes | no | 839.595837 | `6a58a1c385d9643ce16d5fc1` |
| 3 | global | 7 | global | 7 | 99.2188% | yes | yes | no | no | 855.671479 | `6a58a1c385d9643ce16d5fc1` |
| 3 | line_reset | 9 | line_reset | 9 | 99.7396% | yes | yes | yes | no | 860.334133 | `6a58a1c385d9643ce16d5fc1` |
| 4 | global | 7 | global | 7 | 20.3125% | yes | yes | no | no | 853.499591 | `6a58a1d5b1669a49bf077183` |
| 4 | line_reset | 5 | line_reset | 5 | 100.0000% | yes | yes | yes | yes | 844.447764 | `6a58a1d5b1669a49bf077183` |
| 5 | global | 6 | global | 6 | 100.0000% | yes | yes | yes | yes | 850.326022 | `6a58a1d5b1669a49bf077183` |
| 5 | line_reset | 9 | line_reset | 9 | 100.0000% | yes | yes | yes | yes | 846.162587 | `6a58a1d5b1669a49bf077183` |
| 6 | global | 5 | global | 5 | 99.4792% | yes | yes | yes | no | 848.622368 | `6a58a1ebb1669a49bf077185` |
| 6 | line_reset | 7 | line_reset | 7 | 99.4792% | yes | yes | yes | no | 838.877322 | `6a58a1ebb1669a49bf077185` |
| 7 | global | 7 | global | 7 | 25.7812% | yes | yes | no | no | 842.759684 | `6a58a1ebb1669a49bf077185` |
| 7 | line_reset | 9 | line_reset | 9 | 98.1771% | yes | yes | yes | no | 838.539178 | `6a58a1ebb1669a49bf077185` |

## Combined result

- Mean plaintext recovery: **65.5273%**.
- Median plaintext recovery: **98.6979%**.
- Minimum plaintext recovery: **19.7917%**.
- Trials at or above 80%: **9/16**.
- Trials at or above 90%: **9/16**.
- Trials below 40%: **7/16**.
- Mode accuracy: **16/16**.
- Width accuracy: **13/16**.
- Exact permutation recovery: **8/16**.
- Exact plaintext recovery: **3/16**.

## Frozen-gate calculation

| Frozen requirement | Observed | Pass |
|---|---:|:---:|
| Mean recovery ≥80% | 65.5273% | no |
| Median recovery ≥90% | 98.6979% | yes |
| At least 14/16 trials ≥80% | 9/16 | no |
| Mode accuracy ≥14/16 | 16/16 | yes |
| Width accuracy ≥13/16 | 13/16 | yes |
| No recovery below 40% | minimum 19.7917%; 7 failures | no |

**Global gate: FAIL.**

## Decision

The final permitted amendment does not meet the frozen Family T development gate. The solver remains strongly bimodal: nine trials recover at least 80%, while seven trials fall below 40%. Correct mode identification and threshold-level width accuracy do not compensate for the failed mean, trial-count and minimum-recovery requirements.

Family T is closed at development. No locked test is permitted, and no Family T solver may be applied to Voynich data.
