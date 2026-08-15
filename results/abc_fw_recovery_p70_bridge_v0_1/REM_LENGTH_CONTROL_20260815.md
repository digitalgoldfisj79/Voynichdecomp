# Reconstructed ABC-B ReM length control — 2026-08-15

Status: **COMPLETE**.

## VMS exact-null validation
|class|obs|exact null|historical 200-perm null|delta|ratio|
|---|---:|---:|---:|---:|---:|
|short|598|480.833|483.460|-2.627|1.244|
|mid|501|471.362|471.405|-0.043|1.063|
|long|146|110.384|109.880|+0.504|1.323|

## ReM — 50 structure-matched pseudo-corpora
|class|mean obs|mean null|mean ratio|median ratio|p10–p90|fraction ratio>1|
|---|---:|---:|---:|---:|---:|---:|
|short|271.86|396.79|0.685|0.690|0.631–0.733|0.00|
|mid|7.22|27.03|0.267|0.265|0.144–0.382|0.00|
|long|1.10|5.44|0.204|0.194|0.000–0.444|0.00|

Strict short > mid > long ratio gradient: **0.72** of replicates.
Full historical crowding pattern (strict gradient and long <=1.05): **0.72** of replicates.

## Guardrail
This is a frozen reconstruction of the missing historical control, not recovery of the original `rem_matched.py`. The VMS ABC-B criterion remains the primary preregistered decision; this arm tests whether the proposed generic ReM crowding-gradient rationale calibrates.
