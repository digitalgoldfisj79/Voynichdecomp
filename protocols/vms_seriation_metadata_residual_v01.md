# VMS seriation metadata-residual falsification v0.1 — FROZEN BEFORE OUTCOME

Date: 2026-09-07
Protocol ID: `vms_seriation_metadata_residual_20260907_v01`

Question: does the blind Q9–Q10–Q11 neighbourhood survive after removing broad Currier/hand/illustration-category effects from pairwise text distances?

Target fragment was already revealed before this protocol; this is therefore a targeted falsification, not a blind discovery.

For every ZL/RF/IT/GC × LEX/C3/SHAPE channel:
1. compute the original all-pairs Jensen-Shannon distance matrix exactly as in blind seriation v0.1;
2. using ZL reveal metadata only, form a symmetric pairwise design with additive one-hot endpoint membership for `$L`, `$H`, `$I` categories plus same-L, same-H, same-I indicators and an intercept;
3. fit OLS by Moore-Penrose pseudoinverse over all unordered node pairs, with no target-edge weighting or exclusion;
4. residual distance = observed distance minus fitted metadata component;
5. rebuild reciprocal k=2 neighbours on residual distances.

The original blind consensus rule is retained: >=6/12 channels, >=2/3 families, >=3/4 transcriptions.

Q9–Q10–Q11 survives metadata residualisation only if both q09_b67_68↔q10_b69_70 and q10_b69_70↔q11_b71_72 remain consensus candidates. One surviving edge only = PARTIAL. Neither = METADATA_DEPENDENT.

Also report each edge's support out of 12 before and after residualisation. No new total order is optimised.
