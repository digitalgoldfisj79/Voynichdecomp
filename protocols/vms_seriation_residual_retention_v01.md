# VMS seriation residual-retention audit v0.1 — FROZEN BEFORE OUTCOME

Date: 2026-09-07
Protocol ID: `vms_seriation_residual_retention_20260907_v01`

Purpose: apply the already-qualified metadata-residual falsification globally to the 16 stable edges frozen by the blind graph. This test may only delete/degrade pre-existing edges; it may not promote residual-only discoveries.

Frozen stable-edge set:
- q19_b99_102 ↔ q19_b100_101
- q19_b99_102 ↔ q15_b88_89
- q19_b100_101 ↔ q15_b88_89
- q20_b106_113 ↔ q20_b107_112
- q20_b106_113 ↔ q20_b104_115
- q20_b107_112 ↔ q20_b108_111
- q20_b103_116 ↔ q20_b108_111
- q09_b67_68 ↔ q10_b69_70
- q10_b69_70 ↔ q11_b71_72
- q07_b50_55 ↔ q05_b33_40
- q13_b76_83 ↔ q13_b77_82
- q06_b42_47 ↔ q01_b1_8
- q01_b3_6 ↔ q03_b17_24
- q05_b36_37 ↔ q03_b19_22
- q13_b75_84 ↔ q13_b79_80
- q13_b75_84 ↔ q13_b78_81

Residualisation is exactly protocol `vms_seriation_metadata_residual_20260907_v01`: additive `$L/$H/$I` endpoint categories + same-category indicators, OLS over all unordered pairs, then reciprocal k=2 on residual distances.

Retention rule: residual consensus under the original >=6/12, >=2/3 families, >=3/4 transcriptions rule.

A retained multi-edge component is an `ORDERING_SEED_COMPONENT` only if:
- it contains >=3 nodes,
- every retained node degree <=2,
- and at least two of its retained edges were in the original bootstrap-stable set (automatic here).

This label means only a metadata-independent path-shaped neighbourhood; it does not license direction or chronology. Held-out component validation remains mandatory before any serial interpretation.
