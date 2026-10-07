# K4-WFR1 blind-positive replication — 2026-10-07

## RETRACTED / SUPERSEDED FINDINGS

- SUPERSEDED: the earlier 20-case corrected K4c result "passes all five preregistered recovery gates" is not stable under the fresh 40-case blind-positive replication.
- The 20-case result remains historically correct for that panel, but it no longer qualifies the instrument.
- Voynich remains SEALED. Wrong-family specificity was not opened because the fresh blind-positive panel failed one frozen gate.

## Result

Fresh blind-positive panel: n=40.

- median source NMI = 0.6871615540 — FAIL versus frozen >=0.70
- q10 source NMI = 0.6245676628 — PASS versus >=0.60
- NMI >=0.60 in 39/40 = 97.5% — PASS versus >=90%
- median graph-edge F1 = 0.7500000000 — PASS versus >=0.70
- median top-two restart agreement NMI = 0.7402471735 — PASS versus >=0.60

Additional:
- median top-five pairwise agreement NMI = 0.6914663618
- minimum source NMI = 0.5555570539
- NMI >=0.70 in 15/40 = 37.5%
- median oracle NMI = 0.8000866771
- q10 oracle NMI = 0.7317435452

Bounding:
- median-NMI gate effect = -0.012838446
- empirical bootstrap SD of median = 0.008948276
- ratio = -1.435: metric does not resolve whether the population median is truly below 0.70; nevertheless the preregistered qualification gate fails.
- blind minus oracle mean NMI = -0.1106372661
- matched sign-flip null SD = 0.0190505979
- z = -5.808

Exact-normalization QA:
- max scaled finite-difference gradient error = 1.51e-8.

HF shards:
- 6ac64e55df2184ac91ac0c36
- 6ac64e58df2184ac91ac0c38
- 6ac64e5ae7a0dae8a277b816
- 6ac64e5ce7a0dae8a277b818

## Decision

**FAIL_ONE_GATE / TARGET_SEALED**

Do not proceed to wrong-family specificity or Voynich target inversion with this instrument version.

The narrow conclusion is that exact-normalized K4 recovery is real and usually substantial, but current validation-selected blind search does not meet the frozen reliability threshold on a fresh 40-case panel.
