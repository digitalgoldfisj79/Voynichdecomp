# v0.6 Family G — G1 oracle-carrier combined result

Date: 2026-07-16

Status: **PASSED — G2 DEVELOPMENT AUTHORISED**

## Provenance

- Repository: `digitalgoldfisj79/Voynichdecomp`
- Branch: `experiment/terminal-cipher-programme-v0.6-20260716`
- Terminal protocol commit: `b751675e0ffdfa132579feacfe2f0d65f4884479`
- Frozen Family G protocol: `V060_PROTOCOL_FAMILY_G_CARRIER_STEGANOGRAPHY.md`
- G1 implementation commit: `9e5fb7b97f9078748687033441f1c5e350cb3e1f`
- Machine-readable result: `v060_family_g_g1_combined_result.json`
- Combined scientific payload SHA-256: `139c19401fa67b6758430e686d23291cd76a3302f38fdc2f6998cb2baafe4c4c`

## Source shards

| Cover generator | Job ID | Trials | Mean | Minimum | ≥85% | Encrypted mean | Shard SHA-256 |
|---|---|---:|---:|---:|---:|---:|---|
| Markov-2 | `6a58d1b8b1669a49bf077808` | 16 | 96.6146% | 81.2500% | 15/16 | 93.2292% | `42fe86dac275cba5404e48f5276eb2d6d09a59e87e064e62c30f764a248ed811` |
| Motif | `6a58d1c285d9643ce16d63a6` | 16 | 98.7630% | 92.7083% | 16/16 | 97.5260% | `8089700547d898759dba20bfe348657a50ccfab7774ba9678ea4391288dbe4cc` |
| Copy-mutate | `6a58d1c9b1669a49bf07780a` | 16 | 98.9583% | 94.7917% | 16/16 | 97.9167% | `592b9b146f0bffa0790ef1933572315ffd75976e882be2b5d5cf6d96711a12ec` |
| Slot | `6a58d1d185d9643ce16d63a8` | 16 | 98.4375% | 90.6250% | 16/16 | 96.8750% | `296367ab1be534bbccbfdaac785ad19c4955217186a8b888ee9f6002cd706cb9` |

## Combined frozen-gate calculation

- Trials: **64**.
- Mean payload recovery: **98.1934%**.
- Minimum payload recovery: **81.2500%**.
- Trials at or above 85%: **63/64**.
- Encrypted-payload trials: **32**.
- Encrypted-payload mean recovery: **96.3867%**.
- Hidden plaintext-versus-mono status selection: **64/64**.

| Frozen requirement | Observed | Pass |
|---|---:|:---:|
| Mean recovery ≥90% | 98.1934% | yes |
| Minimum recovery ≥70% | 81.2500% | yes |
| At least 58/64 trials ≥85% | 63/64 | yes |
| Encrypted-payload mean ≥85% | 96.3867% | yes |

**Global G1 gate: PASS.**

## Decision

Family G advances to the frozen G2 blind carrier-detection and recovery development stage. This result does not authorise a locked test or Voynich application. The G2 solver must search the complete frozen candidate inventory, calibrate against 256 matched null covers, apply the preregistered multiple-search correction, and either return one carrier rule or abstain.
