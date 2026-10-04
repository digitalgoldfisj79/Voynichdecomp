# Voynich inverse-renderer recoverability contract v1.0
Date: 2026-10-04
Status: FROZEN SPECIFICATION
Scope: synthetic calibration, cipher calibration, and any later Voynich inverse search
Firewall: NO P70

## 1. Principle

No inverse method may be applied interpretively to Voynich unless the exact same search-and-selection pipeline first recovers known hidden inputs generated under a declared forward family.

"Looks plausible", training likelihood, best restart, or a high-scoring latent partition are not recovery.

The scientific object is:

    hidden input X
      -> declared forward transform / renderer F(theta)
      -> observed output Y

The inverse algorithm sees only Y plus the declared model family. It must recover X and/or theta on blinded synthetic/calibration cases.

## 2. Exact forward-model consistency

Every calibration run must record a generator manifest:
- generator family and version;
- corpus/plaintext SHA;
- random seed;
- hidden alphabet/inventory size;
- transition/source model;
- renderer/cipher parameters;
- all support masks and legality constraints;
- termination rules;
- noise/mixing parameters.

Scientific scoring MUST use the same normalized probability family as the generator.

Surrogate objectives (collapsed Dirichlet scores, clustering losses, contrastive losses, spectral scores, heuristic fitness, etc.) are permitted ONLY for proposal generation / initialization.

A surrogate may never be used as the final scientific selection score unless it has independently passed the same recoverability calibration.

All token termination / END decisions that are probabilistic in the generator must appear in the exact likelihood. If END is frozen/non-source-controlled in the generator, the inverse model must treat it identically.

## 3. What counts as recovery

### 3.1 Abstract latent-source experiments
Allowed equivalence:
- arbitrary permutation of latent-state labels.

Primary sequence recovery:
- permutation-invariant NMI;
- ARI;
- posterior/Viterbi agreement;
- transition-graph recovery after optimal state alignment.

PASS for a regime:
- median heldout NMI >= 0.70;
- 10th-percentile heldout NMI >= 0.60;
- >= 90% of calibration replicates have heldout NMI >= 0.60;
- transition-edge F1 >= 0.70 after alignment;
- validation-selected solution, not oracle-selected solution, satisfies these;
- independent search runs converge: median pairwise aligned NMI >= 0.60 among top-decile validation solutions.

A single lucky restart does not pass.

### 3.2 Known cipher calibration
No arbitrary relabelling of decoded plaintext is allowed.

Required outputs:
- plaintext character/token accuracy;
- exact key/transform recovery where the key is identifiable;
- word-level accuracy where tokenization is defined;
- heldout log probability/perplexity;
- edit distance to plaintext.

Only mathematically unavoidable cipher symmetries are allowed and must be declared per cipher family.

PASS:
- >=95% median plaintext symbol accuracy on tractable deterministic cipher families;
- >=90% median on declared noisy/homophonic families;
- >=90% of seeds exceed the family-specific threshold;
- no access to plaintext during model/restart selection.

If a method supplied with the correct cipher family cannot recover known generated plaintext under manuscript-like lengths, it is not licensed for that family on Voynich.

## 4. Calibration ladder

Run in this order. Failure blocks later interpretation.

### C0 Identity / plumbing
Plaintext -> identity renderer.
Must recover exactly (100% except explicitly injected noise).
Purpose: indexing, segmentation, fold and evaluation sanity.

### C1 Monoalphabetic substitution
Random substitution alphabets; multiple plaintext sources and lengths.
Unknown key, known cipher family.
Must recover plaintext and key up to declared alphabet conventions.

### C2 Polyalphabetic / periodic substitution
Known family, unknown period/key within preregistered bounds.
Tests ability to recover stateful mappings rather than static substitution.

### C3 Homophonic substitution
Known family and homophone-budget range.
Tests many-to-one / one-to-many ambiguity and posterior calibration.

### C4 Nomenclator / codebook-like synthetic system
Mixture of ordinary substitution and reusable codebook entries.
Known model family but hidden table.
Tests token/codebook recovery.

### C5 Renderer-native planted source
One hidden source item per Voynich token.
Sparse source dynamics.
Source biases exact 57-piece FORM route decisions through the frozen renderer.
The generator and inverse likelihood are identical.
This is the direct calibration family for the proposed Voynich search.

### C6 Misspecification controls
Generate under each family but fit neighboring WRONG families.
Purpose: prove that the pipeline can reject a wrong family instead of hallucinating a confident decoding.

## 5. Data lengths and difficulty

Every family is tested at:
- N=2,000;
- N=4,000;
- N=8,000;
- N=16,000;
- N approximately 34,000 where applicable.

Difficulty ladders are preregistered before results:
- source inventory K;
- graph sparsity/out-degree;
- renderer/cipher noise;
- homophony;
- key period;
- low-rank coupling strength.

The searchable region is the set of regimes where the full blind inverse pipeline passes recoverability.
We do not extrapolate beyond that envelope.

## 6. Search-selection firewall

For every synthetic replicate:
1. Generate hidden truth and observation.
2. Seal hidden truth.
3. Search using observation only.
4. Use training data to fit candidates.
5. Use inner validation likelihood / exact posterior predictive score to select architecture and restart.
6. Freeze selected model.
7. Open heldout observation and decode.
8. Only then reveal hidden truth and compute NMI/accuracy.

Truth labels may never choose:
- restart;
- architecture;
- stopping point;
- regularization;
- search budget;
- hyperparameters.

Diagnostic use of truth is permitted only AFTER a selection rule is frozen, and must be labelled diagnostic.

## 7. Exact model score

Final candidate ranking uses exact heldout marginal log likelihood under the declared forward model, including:
- source dynamics;
- all source-controlled FORM decisions;
- all probabilistic END decisions;
- base renderer mixture;
- legality masks;
- any declared noise channel.

No collapsed approximation is accepted as the final ranker.

Model complexity is evaluated separately by MDL / marginal evidence:
- transition topology;
- coupling rank;
- source inventory;
- free continuous parameters;
- codebook/table complexity where applicable.

## 8. Predictive equivalence and non-identifiability

Two latent solutions are considered observationally equivalent only if:
- their heldout predictive distributions are statistically indistinguishable;
- their aligned source sequences cannot be separated by calibration truth;
- and their complexity is equivalent within the preregistered tolerance.

State label permutation is always equivalent.

State splitting/merging is NOT automatically equivalent.
It is allowed only when the split/merge leaves predictive distributions unchanged and the minimal model is reported.

If several substantively different latent systems remain predictively equivalent, conclusion = NON-IDENTIFIABLE.
More compute does not select a historical decoding from an equivalence class.

## 9. Best-of-search null

Massive search creates selection bias.

For every real search budget B:
- run the identical algorithm;
- identical number of starts;
- identical successive-halving schedule;
- identical architecture grid;
- identical compute budget

against renderer-only / shuffled / wrong-family controls.

Voynich is compared against the distribution of the BEST solution found in each complete null search.

Per-restart p-values are forbidden.

## 10. Current Phase E-G status under this contract

Phase E collapsed score is a proposal objective, not a valid scientific recovery score for the planted generator because:
- it omits END from source-controlled decisions;
- it integrates emissions under independent Dirichlet context models;
- the planted generator is a bounded low-rank source-to-piece channel with base mixing.

Therefore the observation that the planted partition scored ~909 log units below a wrong partition under the Phase-E collapsed score diagnoses SURROGATE MISMATCH, not failure of the true generative likelihood.

Useful existing diagnostics:
- exact oracle test NMI around 0.76 in the strong piece-channel regime;
- supervised rank-2 fit with planted training labels reaches test NMI ~0.735;
- unsupervised Phase E-G solutions plateau around ~0.46-0.49.

Licensed conclusion:
the representation/channel can carry a recoverable hidden source, but the current unsupervised search/selection pipeline has NOT demonstrated recovery.

## 11. Required next experiment

Build an exact-family blind calibration harness for C5.

Generator:
- current 57-piece FORM support;
- source inventory K=16 initially;
- sparse d=4 source graph;
- rank-2 bounded source coupling;
- exact base-mix/END rules;
- N=4k then 8k/16k/34k;
- strong identifiable regime first (the regime with oracle NMI >=.70).

Inverse:
- exact same likelihood family;
- surrogate methods may propose initial states only;
- final selection strictly by inner validation exact marginal likelihood;
- outer test opened once;
- truth revealed only after selection.

Run >=20 seeds before declaring recovery.

Do not run Voynich and do not benchmark large GPUs until C5 passes this contract.

## 12. Promotion to Voynich

A real Voynich inverse search is licensed only after:
- C0-C1 pass exactly;
- every cipher/source family claimed relevant has passed its own calibration;
- C5 passes at manuscript-like length and complexity;
- C6 demonstrates rejection of neighboring wrong families;
- best-of-search null calibration is operational.

A successful Voynich latent solution then means only:
"the observations support a stable abstract upstream representation under this calibrated family."

It is NOT a plaintext decoding until independent lexical/label/generalization tests pass.
