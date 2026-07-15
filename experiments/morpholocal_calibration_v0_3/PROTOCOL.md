# Morpholocal calibration v0.3 — literature-aligned decoder tournament

**Status:** frozen development protocol

**Parent result:** v0.2 `FAIL_MORPHOLOCAL_CLASS_CALIBRATION`; 34/96 formal positives recovered, 0/320 controls selected; result SHA-256 `c12c48d5585dd4efc5935d29ca2eae46df3c1dabd6475ed89ae6eb7a3c0b1705`.

**Interpretive register:** v0.2 demonstrated failure of one low-order generic annealing decoder. It did not close the bounded historical nomenclator class. V0.3 tests the same bounded class with a preregistered tournament of literature-aligned inference families. It is not a broader cipher hypothesis and does not modify v0.2.

## 1. Primary question

Can a frozen decoder or ensemble reliably recover the existing bounded mixed-unit, homophonic nomenclator class while rejecting matched production-only controls and reconstructing held-out output?

No Voynich manuscript data may be used for model development, decoder selection, language-model extension, threshold setting or formal calibration. No manuscript application is authorised unless every formal gate passes, a clean-clone reproduction agrees, and the hostile audit finds no result-invalidating defect.

## 2. Frozen class boundary

V0.3 retains the v0.2 positive generator and its historical proxy:

- one surface token maps to one latent unit;
- fixed small latent inventory;
- mixed letter/syllable/preposition/whole-word latent units;
- balanced or unequal homophone classes;
- zero or low non-zero null inventories;
- global or Currier-style partitioned keys;
- fixed PGCS-derived surface inventory and legality constraints;
- no creation of new surface forms by the selector;
- iid-uniform, cyclic, frequency-weighted and sticky-line-reset selection policies;
- optional adjacent-length selector;
- held-out plaintext and surface material.

Deferred to a future version: changing keys within a line, unconstrained polygraphs, an independent Voynich-specific production realiser, glyph generation, manuscript-derived vocabulary and any post-inspection expansion of the candidate class.

## 3. Synthetic registry

### 3.1 Positive dimensions

Formal positives must balance or deterministically block across:

- key scheme: global / partitioned;
- null count: zero / low non-zero;
- homophone sizes: balanced / unequal;
- external unit profile: balanced / letter-heavy / word-heavy;
- policy: iid-uniform / cyclic / frequency-weighted / sticky-line-reset;
- selector: none / adjacent-length;
- length: short / medium / long;
- noise: clean / low transcription noise where the common evaluator supports it.

Provisional lengths are 2,000, 8,000 and 36,000 total events with a fixed 80:20 train/test split. Lengths may be altered once during engineering for data-availability or computational reasons, before development results are inspected, and must then be frozen.

Every positive uses a new key, new document allocation and independent policy parameters. Formal keys and documents are disjoint from neural training and development.

### 3.2 Oracle requirement

Every formal positive must pass an oracle precheck using the true key, nulls, structure and policy. Oracle failure is a generator/model-class defect and must be resolved before the formal set is sealed. Oracle accounting is not decoder recovery.

### 3.3 Production controls

Core formal families:

1. context-iid;
2. cell-Markov;
3. copy-mutate;
4. permuted-cipher.

Development additionally includes wrong-language, wrong-register, shuffled-line, surface-mismatch, policy-matched production and neural out-of-key-space controls.

Formal controls are untouched and number at least 160 per core family if compute permits; never fewer than 80 per family.

## 4. Data separation

Three partitions are mandatory:

- deterministic engineering fixtures;
- development benchmark for all tuning and model selection;
- untouched formal benchmark revealed only after the effective-source freeze and hostile pre-run audit.

Formal seeds are stored outside active development until freeze. No formal outcome may influence hyperparameters, ensemble rules or thresholds.

## 5. Common external-model registry

Required scorers:

- v0.2 low-order baseline;
- historical character 3-, 5- and 6-gram models;
- word and subword models;
- combined character-plus-subword model;
- neural historical-language sensitivity;
- matched modern and deliberately wrong-language controls.

Every corpus entry records source, language/register, date range, geography, normalisation, document count, token count, licence and hashes. No Voynich-derived vocabulary extension is permitted. Language/register selection is charged.

When real historical corpora are not already available in the repository, engineering may begin with a clearly labelled synthetic/existing-external-model benchmark, but no formal freeze is permitted until the corpus registry is complete.

## 6. Mixed-unit composition

A shared finite-state/lattice layer must compose recovered latent letters, syllables, prepositions and whole-word units into candidate plaintext streams. It must represent nulls, word boundaries and ambiguous compositions; provide partial path costs to search algorithms; and return latent-unit and reconstructed-character output.

All decoders ultimately report mapping, null set, key scheme, policy estimate/posterior, latent sequence, reconstructed character sequence and confidence.

## 7. Decoder tournament

### A. Specialised heuristic

Nested hill climbing and specialised annealing with separate move families for mapping, class structure, nulls, partition and policy; multiple deterministic restarts; restart-stability and score-margin reporting.

### B. Beam search

Optimised cell assignment order; class-size constrained partial mappings; calibrated or admissible rest cost; higher-order external models; lattice-compatible partial decoding; duplicate suppression and beam-width saturation tests.

### C. Bayesian inference

Multiple independent chains over mappings, nulls, partition and policy; posterior diagnostics, effective sample size, chain agreement, marginal key probabilities and posterior predictive checks. Non-identifiability is reported as unresolved.

### D. Synthetic-trained neural decoder

A recurrence/attention model trained on disjoint random keys and documents. Required safeguards: unseen-key tests, altered key-space geometry, wrong-language controls, out-of-key-space negatives, confidence calibration and direct key/output recovery. Shared dataset-wide homophone pools are prohibited in formal evaluation.

### E. Frozen ensemble

Selected on development only. Candidate rules include unanimous agreement, majority plus minimum output agreement, MDL reranking of pooled candidates and posterior-weighted consensus. “Accept any decoder” is prohibited unless its combined false-positive gate passes.

## 8. Policy-aware inference

Selection policy is a charged candidate or latent mixture, not an unmodelled nuisance. Report within-policy recovery, policy identification, leave-one-policy-out transfer, worst-policy recovery and policy-matched control false positives. A non-identifiable policy is an explicit unresolved branch.

## 9. Accounting

Primary formal convention: fixed-support KT or equivalent preregistered universal/predictive code charging mapping, class structure, nulls, key partition, policy, external register, lattice structure and data.

Secondary reports: `H-key`, `H-fitted`, `I-universal`, held-out predictive cost. No retrospective convention selection. Model preference alone is never sufficient; output recovery is required.

## 10. Development tournament

Progressive stages:

1. exact fixtures;
2. reduced balanced grid;
3. full development grid;
4. frozen decoder/ensemble selection.

A decoder is eliminated for uncontrolled false positives, failure on simple oracle-solvable cases, numerical instability, unacceptable scaling or inability to reconstruct output.

Target development scale: at least 1,152 positives and 640 core controls, with identical trials across decoder families. Progressive subsets are allowed for elimination.

## 11. Effective-source freeze

The freeze hashes every executable component, including generators, patch-free runners, decoder files, lattice, external-model builders, configs, seeds, corpus manifests, artifact schema, dependency lock, container digest and exact command. Runtime source patching is prohibited.

Required records:

- `PROTOCOL.md`;
- `DEVELOPMENT_LOG.md`;
- `DECISION_RULE.json`;
- `THRESHOLDS.json`;
- `CORPUS_REGISTRY.json` and hashes;
- `GENERATOR_MANIFEST.json`;
- `DECODER_MANIFEST.json`;
- `CONTAINER_MANIFEST.json`;
- `ARTIFACT_SCHEMA.json`;
- `FREEZE_MANIFEST.json`;
- sealed formal seeds;
- complete SHA-256 inventory.

A pre-run hostile audit must verify source completeness, data separation, generic gate derivation and the manuscript interlock.

## 12. Formal calibration

Target formal scale: 192 positives and 640 controls (160 per core family). If a balanced complete crossing is impossible, use a deterministic balanced incomplete-block design declared before seeds are opened.

A positive success requires all applicable trial-level conditions:

1. cipher selected under primary accounting;
2. mapping threshold met;
3. null threshold met;
4. key scheme correct;
5. policy correct or posterior sufficiently concentrated;
6. latent-output error below threshold;
7. reconstructed character TER below threshold;
8. held-out predictive non-inferiority;
9. required restart/fold stability.

### Provisional formal gates

- overall positive Wilson 90% lower bound >= 0.70;
- every core positive stratum lower bound >= 0.50;
- every policy stratum lower bound >= 0.50;
- overall false-positive Wilson 90% upper bound <= 0.05;
- every production-family upper bound <= 0.10;
- neural out-of-key-space upper bound <= 0.10;
- median mapping accuracy >= 0.60;
- median null F1 >= 0.75;
- key-scheme recovery >= 0.80;
- policy identification >= 0.75;
- structure recovery >= 0.65;
- median latent-unit sequence error <= 0.20;
- median reconstructed-character TER <= 0.25;
- median word/subword accuracy >= 0.60 where defined;
- held-out key agreement >= 0.70;
- numerical implementation disagreement <= 1e-7 bits;
- exact trial inventory, clean-clone reproduction and independent recomputation.

Output thresholds must be calibrated against oracle and near-oracle fixtures before formal freeze. Any failed gate yields failure or unresolved status. Gates do not compensate for one another.

## 13. Compute and sharding

Prioritise wall-clock time over cost. Use high-core CPU for independent classical trials and Bayesian chains, and high-memory GPUs for neural training, batched scoring and large language models. Formal work is deterministically sharded by positive factorial cell and control family. Every shard is checkpointable and independently reproducible.

Telemetry records job ID, hardware, image digest, commit, command, environment, shard, start/finish times, failures and artifact hash.

## 14. Audits

Required:

- static pre-run freeze audit;
- independent result recomputation of every stratum, interval, criterion and verdict;
- clean-clone reproduction of canonical scientific records;
- hostile model audit for language leakage, neural key-pool shortcuts, policy leakage, target-derived choices, uncharged parameters, search truncation and non-identifiability.

## 15. Decision rule

A manuscript run is admissible only after a complete formal pass, exact or canonical clean-clone reproduction, and hostile-audit clearance. A pass establishes only synthetic recoverability; it does not establish that Voynichese is encrypted.

If v0.3 is conservative but insensitive, the bounded-class route is closed for this tournament and no manuscript run follows. If it is sensitive but non-specific, the decoder is rejected. Policy-specific or neural-only success remains restricted unless its dedicated safeguards pass.

## 16. Execution order

0. preserve v0.2 and create v0.3 branch;
1. build patch-free executable infrastructure and generic gate evaluator;
2. reconstruct benchmark registry and oracle fixtures;
3. build external-model registry;
4. implement mixed-unit lattice;
5. implement four decoders;
6. run progressive development tournament and freeze selection;
7. hostile pre-formal audit;
8. formal calibration;
9. clean-clone reproduction and hostile result audit;
10. explicit manuscript-admissibility decision.

No manuscript data are to be run under this protocol unless step 9 confirms a complete pass.