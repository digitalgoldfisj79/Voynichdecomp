# VSN-B3-v1 — State-Gated K2 Holdout Result

Date: 2026-08-12
Status: **CLOSED — HOLDOUT FAIL**
Binding scientific freeze: `924ef5edd25f884227410dbfad0b59998b33f62f`
Discovery winner frozen before holdout: `LINE-w21-l2`
Discovery-result commit: `bc1f5d3c7c9612eea67fa6139dc1bb790118e639`
Holdout workflow commit: `54a79b7b6794d7f0faf6738c19ab9de50678c08a`
GitHub Actions run: `31576285533`, job `94049033144`
Raw holdout artifact ID: `9133301387`
Raw JSON SHA-256: `82dea727884a2828275aa52fefbb465fbc9053253c2a9d89de8836a5bba6c400`
Artifact ZIP SHA-256 reported by Actions: `110dd12169c8674f9105ccdf0acd6ba0a40f84a7d0e88bac4b3d965315ab1572`
Frozen runner SHA-256: `a534e38d0110c189fe1b008ad53bc844b481344e9af4c41a67aaa2a9fd1cb56c`
Frozen target SHA-256: `2feabfff8ec10ca53c9d3fb4dfdb55f0ce0070104860567a27df237a5ae558ff`

## Pre-holdout discovery result

The exact frozen winner `LINE-w21-l2` reduced discovery mean loss from BASE 6.76550 to 3.86985, a 42.80% improvement, and improved all 4/4 discovery sections. It therefore legitimately passed the preregistered holdout unlock.

No configuration, width, seed, metric, threshold or target was changed after holdout opening.

## Binding holdout result

Overall structural-gating pass: **FALSE**.

### Pharmaceutical

Twenty frozen holdout seeds `2026081301..2026081320`.
Median synthetic metrics vs preregistered criteria:

| metric | target | holdout median | criterion | pass? |
|---|---:|---:|---|---|
| pair ratio | 1.0 by definition | **0.4834605598** | [0.80,1.25] | FAIL |
| edit-location TV | 0 | **0.1937428006** | <=0.08 | FAIL |
| line enrichment | 1.6042402427 | **1.7515715366** | abs diff <=0.25 | PASS |
| H(next\|prev) bits | 2.3076531899 | **3.1581249541** | abs diff <=0.35 | FAIL |
| right-minus-left bits | -0.1165902784 | **+0.0156227928** | negative and abs diff <=0.10 | FAIL |
| mean type length | 5.7491166078 | **4.8860424028** | abs diff <=0.75 | FAIL |

Seed-level all-six pass count: **0/20**.

### Herbal-B

Twenty frozen holdout seeds `2026081301..2026081320`.
Median synthetic metrics:

| metric | target | holdout median | criterion | pass? |
|---|---:|---:|---|---|
| pair ratio | 1.0 by definition | **0.5284191829** | [0.80,1.25] | FAIL |
| edit-location TV | 0 | **0.1414892702** | <=0.08 | FAIL |
| line enrichment | 1.8382454091 | **1.7331400448** | abs diff <=0.25 | PASS |
| H(next\|prev) bits | 2.2457542369 | **3.1017755545** | abs diff <=0.35 | FAIL |
| right-minus-left bits | -0.2401116165 | **+0.0199480666** | negative and abs diff <=0.10 | FAIL |
| mean type length | 5.6979405034 | **4.6887871854** | abs diff <=0.75 | FAIL |

Seed-level all-six pass count: **0/20**.

## Scientific interpretation

The preregistered state/line-gating abstraction can reproduce one real feature on untouched material: **running-text line-local one-edit enrichment**. It does not reproduce the rest of the held-out Voynich architecture.

Specifically, it remains:
- far too sparse in section-local exact one-edit neighbours;
- wrong in edit-location composition;
- much too weakly constrained at first order (`H(next|prev)` too high);
- wrong in the sign of the right-edge positional asymmetry;
- too short at type level.

The discovery improvement was therefore not a general solution to the K2 hierarchy problem. It chiefly learned a generic route to line-local clustering, which transfers, while the harder morphological and positional properties do not.

## Verdict

**FAIL — do not rescue v1 post hoc.**

What remains supported from VSN-B2/B3:
1. Matteo-style K2 composition is a genuine historical mechanism precedent and produces a nontrivial global edit-location topology.
2. Voynich exhibits strong domain/section and line-local selection effects.
3. Simple synthetic context gating can generate the line-local clustering dimension.

What is not supported:
1. literal Matteo K2 as the Voynich generator;
2. this frozen state-gated K2 family as the Voynich generator;
3. assigning Matteo medical states or mnemonic categories to Voynich components;
4. combining Bartolomeo/Ragona rules post hoc to repair the failure.

Any future historical-mechanism model must explain the right-edge/Markov constraints and section-local density **independently**, rather than merely add more gating to this failed family.

## Execution incidents retained

- Hugging Face Jobs status returned repeated upstream HTTP 502 errors before launch; no VSN-B3 HF job was created.
- GitHub Actions was used as the transparent fallback execution substrate.
- First Actions discovery run failed before scoring because shallow checkout could not resolve the immutable freeze commit; this was a transport-only failure.
- Second discovery run fetched the immutable freeze commit explicitly and completed successfully.
- Holdout completed successfully from the exact same freeze.
