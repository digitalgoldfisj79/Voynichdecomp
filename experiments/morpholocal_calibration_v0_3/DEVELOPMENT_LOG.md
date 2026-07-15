# Morpholocal calibration v0.3 — development log

## 2026-07-15: protocol freeze and implementation launch

### Branch and protocol

- Branch: `experiment/morpholocal-calibration-v0.3-20260715`
- Parent v0.2 closure commit: `e203d5f1a69f618297c630fcb30f209accc14343`
- Frozen development protocol commit: `38d130b1242e1542dde759e867fe88f086ae3367`
- The protocol is non-retroactive and preserves the v0.2 failure.
- No Voynich manuscript data are authorised or used.

### Compatibility benchmark inspection

The exact v0.2 generator and evaluator were inspected before v0.3 implementation:

- 24 surface cells;
- 12 payload units, with optional two-unit null extension;
- 18 documents × 2,000 events = 36,000 generated events;
- global and Currier-style keys;
- balanced and unequal homophone classes;
- word-heavy, balanced and letter-heavy external profiles;
- iid-uniform, cyclic, frequency-weighted and sticky-line-reset selection;
- none and adjacent-length surface selectors;
- 52,004-word external historical corpus.

The generator source confirmed that frequency-weighted and sticky selection have explicit context/sequential likelihoods that v0.2 did not infer.

### Classical tournament implementation

- Common tournament runner commit: `383164b6d9fa537c1eba4dba5586f78781d4aa4a`
- Vectorised exact policy likelihood: `c982b11faf199c4536a6f85ae55bef6d5d8e8c93`
- Ordered length-sampler correction: `a1ab71934b256a71ddc3ede80b2b9c5e3e176891`

Implemented development solvers:

1. specialised heuristic search with alternating policy inference;
2. constrained beam search with policy reranking;
3. parallel-tempering Bayesian/Metropolis search;
4. synthetic-trained permutation-equivariant graph decoder.

All classical decoders use common candidate structures, policy-aware likelihoods, charged policy selection, held-out scoring and the four production controls.

### Defect found before scientific pilot

The initial length-subset helper re-sorted events using `token_index`, which is a vocabulary index rather than an event-position index. This could corrupt within-line order. It was identified during smoke testing and corrected before launching the balanced development grid. No scientific pilot result used the defective ordering.

### Classical smoke tests

Technical smoke jobs:

- heuristic: HF job `6a573a6085d9643ce16d383e`;
- beam/Bayesian: HF job `6a573a8885d9643ce16d3840`;
- vectorised three-family smoke: HF job `6a573b4985d9643ce16d3844`.

All interfaces executed and produced complete trial records. Low-budget smoke recovery was deliberately inadequate and is not interpreted scientifically.

### Active classical development jobs

All use seed `3030303`, 96 positives, 64 controls, three length profiles and 24 `cpu-xl` workers.

- heuristic: HF job `6a573bf4b1669a49bf073a00`;
- beam: HF job `6a573c05b1669a49bf073a02`;
- Bayesian: HF job `6a573c17b1669a49bf073a06`.

These are development/model-selection runs, not formal calibration.

### Neural implementation

- Graph decoder commit: `df4e0255813b0ff9aa3592cfb67d1a4185381e32`
- Fork-based development evaluation launcher: `bb9d783ee48b3860782095be0a065bac88222353`

The neural model is permutation-equivariant over surface cells and is trained on independently generated random keys. Cell identities are randomly permuted in every training example. Formal keys and documents are not available to training.

Initial A10G smoke was cancelled while scheduling. T4 smoke job `6a573ee4b1669a49bf073b01` confirmed checkpoint training but exposed dynamic-solver registration loss under spawned evaluation workers. This was an execution defect. The fork launcher was added for development only. Follow-up T4 job `6a573fa2b1669a49bf073b24` completed both training and end-to-end neural evaluation successfully.

Full neural training:

- HF job `6a573f4c85d9643ce16d3912`;
- hardware: A100;
- 6,000 independent synthetic trials;
- 1,500–9,000 generated events per trial;
- 45 training epochs;
- disjoint random keys and random cell permutations.

### Remaining pre-formal requirements

Before any formal freeze:

- complete the compatibility-track tournament;
- implement explicit historical plaintext-unit strings and lattice reconstruction;
- build and hash higher-order historical language models;
- calibrate output-level error thresholds on oracle fixtures;
- select and freeze the individual decoder or ensemble using development data only;
- replace the development v0.2 runtime patch loader with a static effective source;
- create full executable, corpus, container and seed manifests;
- pass the pre-formal hostile audit.
