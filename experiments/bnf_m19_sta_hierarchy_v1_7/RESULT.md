# BnF M19 STA/aaa Transliteration Hierarchy v1.7 — Result

Date: 2026-08-09
Verdict: **STA/AAA INSTRUMENT NOT QUALIFIED**

## Executive result

The programme stopped at the preregistered synthetic qualification gate. No RF H17 or C17 Voynich language score was generated.

The fresh Q3 K=22 qualification identified all six held-out control languages correctly with substantial language margins and perfect independent-fit agreement. Five controls also recovered the exact numerical M19 map at 100%. However the Arabic control recovered only **0.763205** of the occurrence-weighted exact numerical map, below the frozen minimum of **0.85**. Therefore the K=22 qualification gate failed and, because all three representation K values must qualify before Voynich scoring, K=26 and K=36 jobs were cancelled immediately.

## Binding Q3 K=22 gate

- correct language classifications: **6/6** (required 6/6)
- minimum language margin: **0.0899987 nats/unit** (required >=0.05)
- median exact map recovery: **1.0000** (required >=0.95)
- minimum exact map recovery: **0.763205** (required >=0.85) — **FAIL**
- minimum occurrence-weighted independent-fit agreement: **1.0000** (required >=0.90)
- overall gate: **FAIL**

Per-language controls:

| control | top language | margin | exact map recovery | independent-fit agreement |
|---|---:|---:|---:|---:|
| Latin | Latin | 0.101086 | 1.000000 | 1.000000 |
| Italian | Italian | 0.157447 | 1.000000 | 1.000000 |
| German | German | 0.089999 | 1.000000 | 1.000000 |
| French | French | 0.132779 | 1.000000 | 1.000000 |
| Arabic | Arabic | 0.223871 | **0.763205** | 1.000000 |
| Spanish | Spanish | 0.113822 | 1.000000 | 1.000000 |

## Representation census completed before qualification

Frozen RF T17-derived representations were valid and well covered:

- STA family: K=22; H17 retained-character coverage 0.998382.
- Full STA member: K=36; H17 retained-character coverage 0.985886.
- Connected `aaa` unit: K=26; H17 retained-character coverage 0.987046.

The official full-STA -> `aaa` conversion used the pinned `STA-aaa.bit`; it is byte-identical to `STAR-aaa.bit` on RF1b but supports the full STA codes present in IT/ZL/GC.

## Development corrections before binding result

Two earlier qualification attempts were explicitly abandoned and are not evidence:

1. Q1 used an inherited control-span seed namespace; cancelled before any Voynich score.
2. Q2 originally checked 19-value support over the complete 84k control span. The Arabic K=26 span had two normalized `o` characters, both in the 39k held-out portion, leaving the 45k fitting portion unable to emit value 22. Q2 was cancelled before completion and before any Voynich score.

Q3 therefore used a fresh namespace `M19STAv17Q3` and required the **45k fitting half itself** to have plaintext-letter support capable of emitting all 19 BnF values. All binding K=22 control spans passed that support test before fitting.

## What this result means

This is **not** evidence against the BnF M19 hypothesis on Voynich text. It is an instrumentation failure under the preregistered calibration standard: at K=22, the current exact-map recovery requirement is not met for one valid fresh Arabic control, even though language identification remains correct and reproducible.

Because the protocol requires successful numerical-key recovery, not merely correct language ranking, the programme is locked from inspecting RF H17/C17.

The failure is localized: the generalized solver/language discriminator appears strong for Latin, Italian, German, French and Spanish and correctly labels Arabic, but the Arabic M19 numerical key has a non-identifiability or optimization issue sufficient to violate the exact-map recovery gate. Any next iteration must diagnose that on synthetic controls only, under a new fresh qualification namespace, before any Voynich score is permitted.

## Compute closure

After the K=22 gate failed, active K=26 and K=36 qualification jobs were cancelled immediately. No H17/C17 job was launched.
