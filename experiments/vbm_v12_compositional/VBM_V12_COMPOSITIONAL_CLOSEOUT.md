# VBM v12 — Compositional Transducer Synthetic Identifiability closeout

Date: 2026-09-02
Job: `6a97c5d321c5aa7c8364d77c`
Hardware: HF `cpu-upgrade`
Status: **COMPLETED**
Protocol: `VBM_V12_COMPOSITIONAL_TRANSDUCER_PROTOCOL.md`
Pre-binding solver addendum: `VBM_V12_PREBINDING_SOLVER_ADDENDUM.md`
Implementation specification: `VBM_V12_IMPLEMENTATION_SPEC.md`

## Frozen programme verdict

`V12_COMPOSITIONAL_TRANSDUCER_FAILS_PRESSURE_TEST`

Stage A passed every frozen gate and therefore opened Stage B. Stage B passed the overall recovery gates, both source-family gates, and adversarial separation, but failed the stronger frequent-type gate: **3/6** positive replicates passed, versus the preregistered requirement of **>=5/6**. The failed gate remains failed.

No Voynich plaintext was opened or fitted.

## Stage A — V11-scale compact pressure

Dimensions: `KN=16`, `KB=7`, 48 skeletons × 3 e-levels (144 potential nucleus types), 10×9 half-pairs (90 potential bridge types), 2,000 lines, 80/20 FIT/HOLDOUT.

Frozen gate result:

- `REC_ALL >= .90`: **6/6**
- both `REC_N >= .85` and `REC_B >= .85`: **6/6**
- PEAKED family: **3/3** all/key pass
- MODERATE family: **3/3** all/key pass
- frequent-only `.95/.90/.90`: **6/6**
- adversary separation: **PASS**
- median positive HOLD_REGRET: `0.0`
- adversary median regrets:
  - `NUC_BROKEN`: `1.0437683511`
  - `BRIDGE_BROKEN`: `1.2844456570`
  - `BOTH_BROKEN`: `1.6913102839`

Stage-A verdict: **PASS**.

Five of six Stage-A positive replicates recovered the complete induced mapping exactly; the remaining MODERATE replicate recovered `REC_ALL=0.994856`, `REC_N=0.990228`, `REC_B=1.0`, with HOLD_REGRET `0.007191`. Thus the structural transducer was strongly identifiable at the V11-scale surface pressure under the supplied-source oracle.

## Stage B — higher-pressure compositional tail

Dimensions: `KN=24`, `KB=11`, 64 skeletons × 4 e-levels (256 potential nucleus types), 14×13 half-pairs (182 potential bridge types), 4,000 lines, 80/20 FIT/HOLDOUT.

### Positive replicates

| Family | Rep | REC_N | REC_B | REC_ALL | REC_N5 | REC_B5 | REC_ALL5 | HOLD_REGRET | REC_PI | REC_BASE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PEAKED | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.0000 | 1.0000 |
| PEAKED | 1 | 0.945662 | 1.000000 | 0.971398 | 0.945662 | 1.000000 | 0.971398 | 0.056578 | 0.8333 | 0.9219 |
| PEAKED | 2 | 0.903279 | 1.000000 | 0.949070 | 0.903621 | 1.000000 | 0.949259 | 0.346920 | 0.6667 | 0.8281 |
| MODERATE | 0 | 0.853374 | 1.000000 | 0.922807 | 0.853374 | 1.000000 | 0.922807 | 0.156851 | 0.7083 | 0.7969 |
| MODERATE | 1 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.0000 | 1.0000 |
| MODERATE | 2 | 0.896457 | 1.000000 | 0.945509 | 0.896457 | 1.000000 | 0.945509 | 0.052641 | 0.7917 | 0.8750 |

All six positive replicates had `COV_N=COV_B=1.0`. All six recovered the full induced bridge map exactly (`REC_B=1.0`), and all six had gauge-adjusted half-component recovery `REC_HALF_GAUGE=1.0`. The residual failures are therefore entirely on the nucleus/e-operator side, not the half-factorised bridge side.

### Frozen Stage-B gate

- `REC_ALL >= .90`: **6/6** — PASS
- both `REC_N >= .85` and `REC_B >= .85`: **6/6** — PASS
- PEAKED family: **3/3** all/key pass — PASS
- MODERATE family: **3/3** all/key pass — PASS
- adversary separation — PASS
- frequent-only `REC_ALL5 >= .95`, `REC_N5 >= .90`, `REC_B5 >= .90`: **3/6** — **FAIL** (required >=5/6)

The three frequent-only failures were:

- PEAKED-2: `REC_ALL5=0.949259` (just below `.95`), `REC_N5=0.903621`;
- MODERATE-0: `REC_ALL5=0.922807`, `REC_N5=0.853374`;
- MODERATE-2: `REC_ALL5=0.945509`, `REC_N5=0.896457`.

Because this was a conjunctive frozen gate, the Stage-B verdict is FAIL despite the other four conditions passing.

### Adversarial separation

Median HOLD_REGRET:

- positive M12: `0.0546098103`
- `NUC_BROKEN`: `1.4138003828`
- `BRIDGE_BROKEN`: `1.7272431515`
- `BOTH_BROKEN`: `2.2063690917`

The positive model therefore separated all three matched misspecified architectures with substantial margin.

## Interpretation

V12 materially changes the synthetic identifiability picture from V10, but does not cross its own terminal pressure gate.

V10 showed that arbitrary whole-nucleus / whole-bridge dictionaries are non-identifying: radically wrong dictionaries can generate highly plausible held-out language. V12 shows that when the two V11-supported structural ideas are converted into a strongly constrained algebraic transducer, the induced hidden mapping becomes largely recoverable and sharply distinguishable from matched broken-component controls.

The bridge result is especially strong within M12: the factorised boundary map was recovered perfectly in every Stage-B positive replicate, including all raw half components up to gauge. The limiting component is the proposed global e-family permutation operator. At Stage B its raw `pi` recovery ranged from `0.6667` to `1.0`, and induced nucleus recovery ranged from `0.8534` to `1.0`.

This does **not** establish that Voynich uses the M12 algebra. M12 was one deliberately simple synthetic formalisation of the two V11 structural observations. The Stage-B failure also means V12 does not authorise V13 under the preregistered programme.

The scientifically defensible conclusion is narrower:

> The V11 structural constraints are sufficient to transform the VBM inverse problem from grossly non-identifying to strongly but not uniformly identifiable in a synthetic compositional model. Half-factorised boundary structure is highly recoverable; the simple global permutation model for e-family composition remains the bottleneck under increased state/surface pressure.

## Stopping rule

Per preregistration:

- V12 Stage A: **PASSED**
- V12 Stage B: **FAILED**
- V13 Voynich structural test: **NOT AUTHORISED BY V12**
- Voynich plaintext search: **NOT OPENED**

Any continuation must be a fresh programme with a newly justified nucleus-composition hypothesis. The failed Stage-B frequent-type gate may not be relaxed or retroactively redefined.
