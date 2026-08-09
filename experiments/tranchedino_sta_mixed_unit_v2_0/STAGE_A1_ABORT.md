# Tranchedino × STA v2.0 — Stage A1 abort

Date: 2026-08-09
Binding A1 runner: `b10c547a4684cbee800d2b080b12910924588d9b`
Successful transport/execution job: `6a787c873e1f34a7e32c0620`

## Binding event

The two frozen 36-restart T20 optimizer ensembles did **not** converge:

- ensemble A best objective: `-2.49398678501333`
- ensemble B best objective: `-2.4856662191641186`
- occurrence-weighted A/B map agreement: **0.3761903709759385**

The frozen protocol required T20 objective/map convergence before any H20 adjudication. Therefore Stage A1 stopped at the T20 fit gate.

Formal v2.0 status: **T20 FIT NOT CONVERGED / H20 NOT ADMISSIBLE**.

## Runner defect and H20 contamination

The committed runner incorrectly calculated H20 after T20 non-convergence instead of returning immediately. Cancellation was issued after the final T20 log, but the H20 computation completed before cancellation took effect. Consequently H20 is now exposed and must never again be described as held-out/virgin for this programme.

For audit completeness only, the inadmissible diagnostic output was:

- H20 coverage 0.9948868844;
- fixed-map score -2.5715705142;
- Stage-A0 positive-control 5th-percentile floor -2.3672276835;
- within-line shuffle q99 -2.9890038093;
- all four H20 buckets had positive observed-minus-null-median differences;
- runner gate false.

These numbers are **not scientific evidence for or against the hypothesis**, because the prerequisite T20 map convergence failed. They may not be used to tune thresholds or select a map.

## Protected data

RF C20 was not scored. IT/ZL/GC replication streams were not scored. No decoded plaintext string was emitted or inspected.

Any continuation must make optimizer changes using T20 fit behaviour only, freeze those changes prospectively, and use a new held-out panel drawn exclusively from previously unscored C20 folios.
