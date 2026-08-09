# BnF 7342 M19-HMM v0.8 — qualification stop

Date: 2026-08-09
Protocol freeze: `d89367cbc6600141ca06e19b584cd7123f21ca86`
Runner: `9fdec6ae1a9d630bdcb1b6a01c63e7bc63222a17`
HF job: `6a781c3fda2af92a634efe62` — cancelled prospectively after qualification became impossible.

## Verdict

**INSTRUMENT NOT QUALIFIED. NO VOYNICH INFERENCE.**

v0.8 replaced the invalid v0.7 permutation-z language ranking with an exact hidden-letter forward likelihood and used a fresh qualification sentence partition not used in HMM development or LM training.

The first four qualification controls all identified the correct language by held-out forward likelihood:

| target | top language | forward margin (nats/letter) | target mapping accuracy | independent-fit agreement |
|---|---|---:|---:|---:|
| Latin | Latin | 0.11423 | 1.0000 | 1.0000 |
| Italian | Italian | 0.14645 | 1.0000 | 1.0000 |
| German | German | 0.08557 | 1.0000 | 1.0000 |
| French | French | 0.14048 | 0.89123 | **0.89082** |

The frozen gate required **minimum independent-fit agreement >=0.90** across all six controls. French therefore made Q5 impossible after the fourth control. The job was cancelled before Arabic/Spanish and before any Voynich stage to avoid unnecessary compute.

The French result is diagnostically useful: language identification itself is strong and correct, but the pairwise numerical-key optimizer has not demonstrated the required reproducibility. The appropriate next step is to improve mapping convergence on controls only and then use a new untouched qualification partition; lowering the 0.90 threshold post hoc is not permitted.
