# Cipher-versus-generated model comparison v0.7

Date frozen: 2026-07-16

Branch: `experiment/cipher-generated-mdl-v0.7-20260716`

Status: **FROZEN BEFORE V0.7 DEVELOPMENT OR LOCKED RESULTS**

## 1. Purpose

V0.6 established that recovery, family recognition and manuscript transfer are separate evidential links. One changing-alphabet construction was recoverable, while the blind family detector did not generalise. V0.7 therefore does not add another Voynich solver. It tests whether a charged source-message model can be distinguished from structured non-message generation by held-out predictive transfer.

The target claim is deliberately narrow:

> A cipher explanation is supported only when a mapping and source model learned on training documents compress independent held-out documents better than production-only models after charging all key, structure, policy and model-selection costs.

Latent order, fluency, decoder confidence and in-sample fit are not sufficient.

V0.6 remains closed and unchanged.

## 2. Literature and empirical motivation

The programme is motivated by three established observations:

1. real historical ciphers can be materially harder than matched synthetic ciphers;
2. source-language and ciphertext-type identification are distinct from decipherment;
3. minimum-description-length model comparison must charge both fit and model complexity.

Initial anchors include Aldarrab and May (2021, 2022), Kambhatla, Born and Sarkar (2023), the DECODE database, and Grünwald and Roos on MDL.

## 3. Stage A: generator-disjoint source-transfer benchmark

### 3.1 Fixed data-generating benchmark

Stage A reuses the independently written v0.3.4 generator-disjoint construction, not the earlier detector-compatible positive generator.

Positive source-message mechanisms:

1. keyed-PRF homophony, global key, no nulls;
2. rotor homophony, Currier-split keys, no nulls;
3. feedback homophony, global key, two null cells;
4. line-keyed homophony, Currier-split keys, two null cells.

Source corpora:

- `greek_corpus_parsed.pkl`;
- `greek_dmm_corpus.pkl`.

Structured non-message controls:

1. ordered hidden Markov process;
2. finite motif grammar;
3. document-topic finite-state process;
4. copied-line process with mutation.

Controls use the same 24-cell surface registry, key-size geometry and surface renderers as positives. They therefore cannot be rejected merely because they lack latent order or cipher-like emission statistics.

Each trial contains 12 documents and 180 surface tokens per document. Complete documents, not random tokens, are assigned to training and held-out partitions. No document crosses partitions.

### 3.2 Development and locked inventories

Development:

- 32 positives: 4 mechanisms × 2 corpora × 4 replicates;
- 32 controls: 4 families × 8 replicates.

Locked test:

- 64 positives: 4 mechanisms × 2 corpora × 8 new replicates;
- 64 controls: 4 families × 16 new replicates.

Development and locked seeds are disjoint and deterministic. One development amendment is permitted. No locked-test amendment is permitted.

### 3.3 Source-model registry

The source registry is constructed without Voynich data.

For each 12-unit stream, candidate predictive models are:

1. a leave-target-corpus-out universal KT bigram model built from the other Greek corpus and the six pinned v0.5 UD corpora after the frozen 12-class word-shape transform;
2. a leave-target-corpus-out universal KT trigram model from the same sources;
3. a pooled multilingual KT bigram model;
4. a pooled multilingual KT trigram model.

The true target corpus is excluded from the primary leave-target-out candidates. Pooled candidates are retained as secondary sensitivity analyses and pay the same registry-index charge. The candidate index costs `log2(K)` bits.

No source model may be extended with decoded trial output or Voynich vocabulary.

### 3.4 Cipher/source-message model

For each trial, the cipher arm must infer from training documents only:

- global versus Currier-partitioned key;
- zero versus two null cells;
- balanced versus unequal homophone-class sizes;
- mapping from 24 surface cells to 12 payload units plus optional null;
- emission policy from the frozen policy registry;
- source model from the frozen source registry.

Inference uses the existing beam-plus-refinement mapping solver, but the objective is replaced by the frozen source-model registry. The mapping, structure, policy and source-model choice are then frozen.

The held-out codelength is the sum of:

- source predictive bits for decoded held-out units;
- emission-policy predictive bits for held-out surface choices;
- selector or token-realisation predictive bits where applicable.

The two-part total additionally charges:

- mapping and partition description;
- null and class-size structure;
- policy index;
- source-model index;
- any selector index.

### 3.5 Production-only model

The production comparator is the frozen charged registry:

1. context-iid;
2. cell-Markov;
3. context-cell-Markov;
4. repeat-context.

The production model is selected on training documents only, pays `log2(4)` bits, and scores held-out documents with frozen training probabilities.

### 3.6 Decision rule

A trial is classified as source-message/cipher only if all conditions hold:

1. total two-part codelength advantage over production is at least `0.05` bits per all token;
2. held-out predictive advantage is at least `0.02` bits per held-out token;
3. the cipher arm wins under both full and conditional accounting;
4. the selected source model is a leave-target-out model, not a pooled sensitivity model;
5. two deterministic training-document folds produce equivalent mappings after optimal latent-label permutation with agreement at least `0.70`;
6. the held-out advantage has the same sign in both document folds.

For synthetic positive evaluation only, success additionally requires:

- mapping accuracy at least `0.55` after optimal label alignment;
- latent held-out unit error at most `0.35`;
- key partition and null-count recovery correct.

Truth is used only for evaluation after the decision is frozen, never for fitting or selection.

### 3.7 Stage A gates

All must pass on the locked set:

- overall positive sensitivity at least `0.70`;
- every positive mechanism sensitivity at least `0.50`;
- each source corpus sensitivity at least `0.60`;
- overall control false-positive rate at most `0.10`;
- every control-family false-positive rate at most `0.20`;
- median positive held-out predictive advantage greater than zero;
- median control held-out predictive advantage less than or equal to zero;
- no evidence of train/test document leakage or target-corpus leakage.

Failure closes v0.7 before real historical validation. Compute, thresholds and model capacity may not be increased after a locked failure.

## 4. Stage B: real historical and notation validation

Stage B is authorised only if Stage A passes.

The external registry must contain, subject to licence and transcription availability:

- at least 12 real historical ciphertexts with known or independently established keys/plaintexts, including homophonic or changing-alphabet examples where available;
- at least 12 real non-cipher controls drawn from abbreviated, tabular, recipe, calendrical, astronomical or other conventional notation;
- document-level partitions that prevent fragments of one manuscript from crossing train and test;
- explicit provenance, licence, date, language, transcription convention and hashes.

The DECODE database and HistCorp are preferred sources. Database access limitations or unsuitable licences are reported as an unresolved data constraint; synthetic substitutes do not count as Stage B.

Required Stage B gates:

- cipher sensitivity at least `0.70`;
- notation/control false-positive rate at most `0.10`;
- leave-one-manuscript-out performance at least `0.60` sensitivity and at most `0.15` false positives;
- no source may dominate the result;
- all decisions use the unchanged Stage A-locked accounting and thresholds.

Failure or inability to construct the required real-data panel blocks Voynich application.

## 5. Stage C: sealed Voynich comparison

Stage C is authorised only after Stages A and B pass and a clean-clone reproduction agrees.

The application uses at least three independent EVA transcriptions, true line boundaries, deterministic held-out folio folds and section-level reports. It reports only codelength differences, stability and abstention. No plaintext is generated unless an independently specified cipher representation supplies the missing structural information required by its solver.

A positive cipher result requires:

- cipher/source-message advantage across all three transcriptions;
- positive held-out advantage in both folio folds;
- positive evidence in at least four adequately represented sections;
- stable mapping structure across folds and transcriptions;
- advantage over the strongest frozen morpholocal/production generator;
- no manual glyph merging, ring ordering, semantic selection or post-result adjustment.

Otherwise the result is abstention or production-preferred.

## 6. Reproducibility and stopping discipline

Required before each locked stage:

- complete source and corpus hashes;
- exact command and container record;
- deterministic seeds stored before execution;
- finite-state and runtime preflight;
- unit tests for split disjointness, accounting identities and label-permutation alignment;
- result SHA-256 and immutable job identifier;
- clean-clone recomputation of the verdict.

Invalid execution may be corrected only when the defect is demonstrated independently of the outcome and the scientific objective, data, thresholds and budgets remain unchanged.

No new comparator, threshold relaxation or family proliferation is permitted after a locked negative result.
