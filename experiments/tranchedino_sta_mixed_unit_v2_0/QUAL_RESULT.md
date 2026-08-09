# Tranchedino × STA v2.0 — Stage A0 Qualification Result

Date: 2026-08-09
Protocol freeze: `3074c2c2f9922290c915034e33ef164d7629ee24`
Implementation amendment: `cbac5abec3dd284fa9181955f8287e94a5069406`
Runner: `7cb98b13f47ca5f3b5b81b05c67654023a6f0018`
Recovered historical archive SHA-256: `ddae949a2d4ff13714204f3751feaf9e836333ef57a45def77c803cd87fc7b61`

## Source reconstruction

The recovered old Paduan split reproduced:

- language-model partition: 4,119 retained lines / 172,347 normalised 19-letter characters;
- held-out payload partition: 1,423 lines / 54,750 characters;
- chronological cut page: 183.

Minor count differences from the old summary's 172,362/54,764 arise from applying the explicitly frozen 19-letter normaliser in the v2.0 runner rather than counting all earlier `cipher_letters`; the source files and split rule are unchanged.

## Historical geometry

The primary key is the real Tranchedino f069v grid: 36 strict signs across 19 plaintext columns, with multiplicities 1 for `a,c` and 2 for every other retained letter. This exactly matches the already-frozen full-STA K=36 surface vocabulary size and was selected before any Voynich scoring.

## Fresh-control qualification

Twelve fresh 12,000-letter held-out Paduan controls were generated with independent opaque f069v-geometry keys and solved blind by independent A/B optimizer ensembles.

Every control converged at the minimum six restarts per ensemble.

| Metric | Result | Frozen gate |
|---|---:|---:|
| converged controls | **12/12** | 12/12 |
| median plaintext recovery | **1.0000** | >=0.95 |
| minimum plaintext recovery | **1.0000** | >=0.85 |
| median weighted true-map recovery | **1.0000** | >=0.95 |
| minimum weighted true-map recovery | **1.0000** | >=0.85 |
| minimum A/B map agreement | **1.0000** | >=0.90 |
| maximum true-map objective advantage over recovered map | **0.0 nats/event** | <=1e-5 |

For all 12 controls the best recovered objective was exactly the hidden true-key objective at reported precision.

## Verdict

**TRANCHEDINO-STA INSTRUMENT QUALIFIED.**

The exact historical 36-sign Tranchedino geometry is strongly recoverable on fresh, held-out Paduan ciphertext at the planned data scale. Under the frozen protocol, RF T20 fitting and H20 adjudication are therefore unlocked. No Voynich decoded text has been inspected.
