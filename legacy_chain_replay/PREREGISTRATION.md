# VOYNICH-LEGACY-CHAIN-REPLAY-v0.1 — preregistration

Date: 2026-08-14

## Question

Does the newly isolated, RF/STA/connected-aaa-robust `ed1_chain_lift` residual require a new mechanism, or is it already an unmeasured consequence of a previously tested generator?

## Freeze

Protocol SHA-256: `d464cdc717e55d4233e2e5700be85b14fa2bc62a7691ac024b9e9bf98949533f`.

No numeric native-EVA Voynich chain-lift target is to be computed before this protocol, hash, runner and target-free QA are committed. The workflow file is committed last.

## Representation discipline

The Voynich residual was discovered independently in RF-member, STA-family and connected-aaa representations. Legacy generators emit EVA-like strings. They will therefore be scored in their native character space. No synthetic EVA→STA/aaa mapping will be invented. The RF/STA/aaa result qualifies the *target phenomenon* as representation-robust; it is not a licence to fabricate alternate representations of legacy output.

## Primary observable

`ed1_chain_lift = P(ED1_{i+1} | ED1_i) - P(ED1)` within physical lines, where ED1 is character Levenshtein distance exactly one and exact-equal tokens are excluded from ED1.

Secondary metrics are diagnostic only: chain rate, ED1 lags 1–4, exact lag-1, adjacent length coupling, ABA and ED1 return.

## Frozen primary legacy arms

1. **G′ regional working-set model** — specification reconstruction from the July 8 record with `R=64`, `beta=8`, `s=.186`, `mu=.0009`; seeds 1–20. No parameter changes.
2. **Faithful Timm–Schinner default** — exact published executable at commit `a6ede2202dd7ad6285ce2c007bf22c2a0e7709b7`; seeds 1–20.

Frozen Timm ablations (`noreuse`, `random`, `position`) are causal diagnostics only and cannot win the primary adjudication.

Q57b is included only if its exact archived dependency stack can be recovered. If `q56_injective_anonymous_realiser.py` is unavailable, the arm is recorded `NOT_EXACTLY_REPLAYABLE`; a later generator must not be substituted.

## Controls

- 500 target within-line shuffles.
- 50 within-line shuffles per generated seed.
- 20 line-order shuffles as an exact invariance QA.

## Adjudication

For each primary legacy arm, let `M` be median chain lift across frozen seeds and `T` the unopened native-EVA Voynich value.

`MODEL_MATCH` requires all three:

1. same sign as `T`;
2. `abs(M/T - 1) <= 0.25`;
3. `M` exceeds the pooled within-line-shuffle null by `z >= 2`.

`MODEL_PARTIAL` requires same sign and `abs(M/T - 1) <= 0.50` but fails the full match.

Final result:

- `CHAIN_LIFT_ALREADY_IMPLIED_BY_LEGACY_MECHANISM` if G′ or faithful Timm default is `MODEL_MATCH`;
- otherwise `CHAIN_LIFT_NEW_MECHANISTIC_GAP_WITHIN_TESTED_LEGACY_ARMS`.

Q57b availability cannot determine this verdict and must be disclosed separately.

## Prohibitions

No parameter search, no choice of winning ablation, no retuning after target opening, no pseudo-STA conversion, no substitute for missing Q57b dependencies, and no semantic/historical attribution from a statistical match.
