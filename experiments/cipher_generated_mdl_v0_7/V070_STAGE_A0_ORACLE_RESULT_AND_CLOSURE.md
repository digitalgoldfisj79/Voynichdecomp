# v0.7 Stage A0 oracle source-transfer result and programme closure

Date: 2026-07-16

Verdict: **ORACLE SOURCE-TRANSFER GATE FAILED. V0.7 IS CLOSED.**

No inferred-mapping development grid, locked Stage A test, real-historical Stage B panel or Voynich Stage C application was run.

## 1. Question tested

The oracle precheck asked whether a charged source-message model can distinguish independently enciphered language from strongly ordered non-message generation when the true surface-to-latent mapping, key partition and null structure are supplied.

This removes cryptanalytic search failure from the test. If the comparator fails here, additional mapping search, GPUs or restarts cannot make the source-transfer criterion valid.

The true mapping was still charged in the two-part codelength. Source model, source order, emission policy and selector were chosen from training documents only. Complete documents were held out.

## 2. Frozen inventory

- 16 source-message positives;
- 16 ordered non-message controls;
- 12 documents and 180 surface tokens per document;
- four positive renderers: keyed PRF, rotor, feedback with nulls, and line-keyed with nulls;
- two independent Greek source corpora;
- four ordered control generators: HMM, motif grammar, topic FSM and copy-mutate;
- identical 24-cell surface registry and renderer geometry for positives and controls.

Protocol: `V070_STAGE_A0_ORACLE_FREEZE.md`.

## 3. Valid execution

Primary valid job:

- Hugging Face job: `Digitalgoldfish79/6a594b54b1669a49bf0785a6`;
- Git head: `d8b2a55858fb5d03cd052e447b9edf4a56330ad7`;
- hardware: `cpu-xl`;
- running time: 42 seconds;
- emitted result SHA-256: `976903a57d6c1da942d16fd5ec34c922e51d681c04c3a85f6998b008f4fa082f`.

Exact clean-clone diagnostic recomputation:

- Hugging Face job: `Digitalgoldfish79/6a594ba485d9643ce16d6be7`;
- Git head: `3b8e08e4b83c8597570094c41a975127c6337287`;
- hardware: `cpu-xl`;
- running time: 42 seconds;
- recomputation result SHA-256: `f9c164e4ef040b2850667e883d47b33ccaa8442e03e7115c53f9f7564e279cd3`.

The complete verdict, trial classifications, aggregate rates and numeric codelength outputs agreed. The differing emitted hashes are a reporting defect: elapsed wall-clock time was included inside the hashed payload. This prevents byte-identical hashes across otherwise identical recomputations but does not alter any scientific value or verdict. The defect is retained and disclosed rather than retrospectively rewriting the primary hash.

## 4. Pre-result execution defects

Three defects were found before a valid oracle score existed:

1. the inherited solver expected its `ExternalModels` dataclass but received a plain dictionary;
2. forked workers raced while initially populating the shared UD corpus cache;
3. the inherited topic-FSM control could jump outside its topic set and then call `topic.index()` on the out-of-topic state.

Corrections were execution-only:

- construct the required immutable `ExternalModels` instance;
- build and hash-verify corpus/model assets in the parent before forking;
- make an out-of-topic topic-FSM state re-enter the fixed topic set before resuming the registered cycle.

No objective, model registry, threshold, trial, seed or gate was changed. The failed jobs produced no aggregate scientific result.

## 5. Registered oracle result

| Metric | Result | Required | Outcome |
|---|---:|---:|---|
| Positive sensitivity | **0/16 = 0%** | >=75% | Fail |
| Ordered-control false-positive rate | **0/16 = 0%** | <=15% | Pass |
| Median positive held-out gain | **+0.166788 bits/token** | >0 | Pass |
| Median control held-out gain | **-1.417854 bits/token** | <=0 | Pass |
| Positive mechanisms represented | **0/4 passed decision** | each >=1/2 | Fail |
| Source corpora represented | **0/2 passed decision** | each >=62.5% | Fail |
| Control families falsely accepted | **0/4** | each <=1/4 | Pass |

The frozen gate failed and returned `STOP_V070_ORACLE_FAILED`.

## 6. Why every positive failed

The result was not a simple absence of signal.

With the true mapping supplied, every positive achieved a large favourable total two-part codelength difference. The source-message arm's total advantage ranged from approximately **1.85 to 2.06 bits per token**, while all four ordered control families were rejected.

However, every positive selected the pooled trigram source model. None selected the required leave-target-corpus-out source model. Therefore all 16 failed the primary transfer condition.

The pooled model included the source corpus family being evaluated. It is useful as a sensitivity analysis but cannot support transfer to an unknown document such as Voynich. Allowing it as primary would convert corpus familiarity into apparent source-message evidence.

The held-out predictive result was also mechanism-dependent:

| Positive renderer | Trials with held-out gain >=0.02 | Trials |
|---|---:|---:|
| Rotor | **4** | 4 |
| Line-keyed with nulls | **4** | 4 |
| Feedback with nulls | **1** | 4 |
| Keyed PRF | **0** | 4 |
| **Total** | **9** | **16** |

Thus, even if the leave-target-out safeguard were discarded after seeing the result, only **56.25%** of positives would clear the held-out-gain threshold. That remains below the frozen 75% oracle requirement.

## 7. Interpretation

The experiment separates three phenomena:

1. **In-sample explanatory compression.** Given the true mapping and a closely matched pooled source model, the cipher/source arm compresses positive trials much better than the production registry.
2. **Specificity against current structured controls.** All 16 ordered non-message controls were correctly rejected.
3. **Out-of-source transfer.** The advantage did not transfer reliably when the exact source corpus was withheld, and held-out performance varied sharply by surface renderer.

The source-message signal is therefore real but conditional on source-model proximity and renderer. It is not a general detector of an encoded message.

This reproduces the central v0.3.4 lesson in a stronger form. V0.3.4 showed that latent order was non-specific. V0.7 shows that even true-map MDL comparison can be conservative and highly source-dependent: it may distinguish some enciphered source/rendering combinations while failing others generated from genuine language.

## 8. Scientific consequence

Under the frozen stop rule:

- no inferred-mapping Stage A development is permitted;
- no Stage A locked test is permitted;
- no real historical cipher/notation Stage B panel is permitted under this implementation;
- no Voynich data may be scored;
- no threshold relaxation, pooled-model promotion or post-result source-registry expansion is permitted;
- additional compute is scientifically irrelevant to this closed comparator.

V0.7 does not show that cipher and generated text are universally indistinguishable. It shows that this charged source-transfer implementation cannot establish that distinction with adequate sensitivity under source and renderer shift, even when the true cipher mapping is supplied.

## 9. Programme-level message

The cumulative result is now:

- some bounded cipher constructions are not reliably recoverable;
- one changing-alphabet construction is highly recoverable when its family and representation are known;
- family identification does not reliably transfer to unseen structural regimes;
- latent order is not specific to a source message;
- charged source-message compression can be highly specific but insufficiently sensitive and dependent on a closely matched source model;
- therefore neither more search nor more compute resolves the present evidential bottleneck.

The remaining barrier is external anchoring: real historical cipher panels, genuine human/medieval non-message corpora, independently constrained representation assumptions, or physical-manuscript evidence. Those are new data programmes, not permissible amendments to v0.7.
