# BnF M19 STA/aaa v1.9 — Binding Qualification Result

Date: 2026-08-09
Protocol freeze: `67ff4e032395030d18b84a630c216e55854ff3a3`
Runner: `bf50f1408acb9614294f4d58b910a7645f24417a`
Namespace: `M19STAv19Q1`

HF jobs:
- K=22: `6a786847da2af92a634f017b`
- K=26: `6a7868603e1f34a7e32c042b`
- K=36: `6a786875da2af92a634f0181`

All six fresh positive controls were rerun independently at each K. Each fit used two independent adaptive optimizer ensembles and was required to converge both by objective and occurrence-weighted map agreement. Synthetic-control fits also had to reach the known true-map objective basin; this oracle is not available or used on Voynich.

| K | correct | min language margin | median map recovery | min map recovery | min A/B agreement | all converged | min oracle gap | Gate |
|---|---:|---:|---:|---:|---:|---|---:|---|
| 22 | 6/6 | 0.090706 | 1.000000 | 0.965641 | 1.000000 | yes | 0.0 | **PASS** |
| 26 | 6/6 | 0.088831 | 1.000000 | 1.000000 | 1.000000 | yes | 0.0 | **PASS** |
| 36 | 6/6 | 0.089087 | 1.000000 | 0.982179 | 1.000000 | yes | 0.0 | **PASS** |

The previous v1.7 K=22 Arabic failure does not reproduce under the stronger optimizer. On fresh v1.9 Arabic K=22 data, Arabic ranks first by 0.233596 nats/letter, exact occurrence-weighted map recovery is 0.965641, A/B agreement is 1.0, and both ensembles converge after the minimum six restarts to within +4.34e-7 nats/event of the hidden-map objective.

## Formal qualification verdict

**STA/AAA STRONG INSTRUMENT QUALIFIED.**

All three frozen representation scales pass without lowering any v1.7 map-recovery or language-margin threshold. Under the frozen v1.9 protocol, RF H19 language scoring is therefore unlocked. RF C19 remains sealed until all three H19 representations independently pass and select the same top language.
