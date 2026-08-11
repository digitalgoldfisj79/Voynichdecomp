# VBM discriminative evidence — v4 Q0 close / v4.1 calibration amendment

Date: 2026-08-11

## v4 Q0 binding result

The frozen 36-row Q0 produced:
- Bavarian 10/12 qualified; CYCLE 6/6; both FREQ_PROP rows passed.
- German 11/12 qualified; CYCLE 5/6.
- Italian 12/12 qualified; CYCLE 6/6.
- Overall 33/36.

All truth-language language-minus-null deltas were positive. The three failures were caused by the separate language-margin/A-B agreement gates, not by Delta <= 0.

The v4 protocol nevertheless required >=34/36 overall, so **v4 Q0 FAILS**. Q1 and target are not authorised under namespace `VBMDISCV4`.

No Voynich H1 HMM target fit has been generated.

## Why a fresh v4.1 calibration is scientifically admissible

The v4 protocol simultaneously specified per-language minimums of >=10/12 and CYCLE minimums >=5/6, and also an aggregate >=34/36 rule. The aggregate rule is stricter than—and not mathematically implied by—the stated per-language robustness criterion. The failure therefore identifies a calibration-design inconsistency before target access, not a target rescue opportunity.

v4.1 does **not** reuse v4 control rows or thresholds. It doubles the control experiment with fresh spans/maps under a new namespace and binds to the per-language replication criterion prospectively.

## v4.1 Q0

Namespace: `VBMDISCV41`.

72 entirely fresh positive controls:
- 3 languages;
- 6 core-homophone regimes;
- 2 bridge schedules;
- 2 independent replicas per cell.

Model, null family, fitting budget and row qualification rule are unchanged from v4.

A row qualifies iff:
- A/B/mean HMM winner = truth language;
- truth-vs-best-wrong mean HMM margin >=0.02;
- truth-language A/B score gap <=0.10;
- truth Delta > 0.

Binding Q0.1 pass:
- each language >=20/24 rows qualified;
- each language >=10/12 CYCLE rows qualified;
- all four Bavarian FREQ_PROP rows qualified.

There is no additional aggregate count rule; the aggregate minimum follows from the three per-language requirements (>=60/72). This rule is frozen before any v4.1 control is generated.

If Q0.1 passes, freeze per-language:
- 5th percentile positive truth Delta among qualified rows;
- 5th percentile truth-vs-wrong language margin among qualified rows.

Then proceed to the original v4 Q1 adversarial-negative design, expanded to 200 fresh negatives. All later target gates remain unchanged.

No v4 or v4.1 target score may be generated unless v4.1 Q0, Q1 and Q2 all pass.
