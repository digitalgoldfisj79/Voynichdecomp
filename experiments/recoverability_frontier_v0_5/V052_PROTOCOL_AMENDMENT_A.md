# Recoverability frontier v0.5.2 — protocol amendment A

Date: 2026-07-15

Status: fixed after the initial smoke and oracle-inventory diagnostic, before any full six-language run.

## Diagnostic result

The initial fixed-inventory smoke failed its test gate:

- mean recovery: 57.2049%;
- mean inferred inventory overlap: 87.1169%;
- English: 62.3264%;
- Turkish: 52.0833%.

A 24-trial development ablation using the true observed homophone-label multiset achieved 72.9601%, versus 65.2778% with the inferred inventory. The 7.6823-point oracle gain shows that inventory inference is material, but the oracle arm is not perfect, so key search also remains limiting.

## Frozen correction

The next development search may alter the inferred plaintext-label multiset rather than preserving it exactly.

Allowed state moves:

1. swap the plaintext assignments of two observed cipher symbols;
2. reassign one cipher symbol to another plaintext label;
3. globally restart from a random bounded homophone-slot subset.

Hard constraint:

- no plaintext label may receive more cipher symbols than the benchmark family's known multiplicity cap.

The solver still receives no key, true inventory, raw surface-symbol integers or test plaintext. The language model remains train-only, and schedule selection remains development-only.

## Gate

The full six-language run remains prohibited unless the corrected English/Turkish 96-character smoke reaches:

- at least 70% overall mean recovery;
- at least 60% in each language;
- improvement over the fixed-inventory test result.
