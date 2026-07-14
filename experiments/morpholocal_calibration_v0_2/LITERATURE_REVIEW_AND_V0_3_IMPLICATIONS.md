# Targeted literature review: bounded morpho-local nomenclator recovery

Date: 2026-07-14

Status: literature review completed after the v0.2 CPU development run. It does not alter the v0.2 generator, thresholds, decoder, or result. Any methodological changes described below belong to a new version.

## Question

Is the v0.2 programme—a mixed-unit, homophonic/nomenclator cipher calibrated on synthetic positives and matched production-only controls, with held-out transfer and charged model comparison—consistent with established historical-cryptology and statistical-model-selection practice? What does the literature imply about the meaning of the 32/96 CPU recovery result?

## Executive conclusion

The historical cipher class is strongly grounded. Mixed letter, syllable, preposition, word/name, homophone, null, and variable-length elements are documented features of European nomenclators. The synthetic-recovery requirement is also well grounded: published decipherment systems are routinely evaluated on generated ciphers with known plaintext/key structure and then applied to real historical material.

The current v0.2 *decoder*, however, is not a literature-standard general solution to that class. It uses one low-order external transition model and one generic annealing objective to solve, simultaneously, key assignment, global-versus-Currier structure, null count, class-size profile, external-register selection, and surface-selection effects. Published work normally reduces this burden by using one or more of the following:

- higher-order character language models;
- word dictionaries or word/subword models;
- beam search with strong rest-cost estimates and an optimized symbol order;
- nested hill climbing or specialised simulated annealing;
- Bayesian sampling;
- explicit segmentation/decoding lattices;
- supervised synthetic training over a defined key space;
- separate treatment of cipher classification, segmentation, key recovery, and plaintext scoring.

Therefore the CPU result—32/96 planted recoveries with 0/64 false positives—is a valid failure of the frozen v0.2 decoder and gate, but it is not evidence sufficient to close the bounded historical cipher class. The oracle result (all accounting signs correct in 96/96; 92/96 predictive non-inferiority) and learned-decoder result together locate the current bottleneck primarily in inference/search and model misspecification rather than in representability.

The matched production controls and the requirement that recovered plaintext transfer to held-out material are stronger than most published historical decipherment evaluations and should be retained.

## 1. Historical grounding of the candidate class

Megyesi et al. analyse more than 1,600 early-modern European cipher keys from ten countries and report increasing diversity in symbol sets, code lengths, code types, nomenclature size, and the linguistic entities encoded. This directly supports treating the historical key as a mixture rather than as a simple letter substitution [1].

Aldarrab and May give an operational taxonomy highly relevant to this programme. Their historical cipher elements include:

- regular elements encoding letters, common syllables, or prepositions;
- nomenclature elements encoding whole words, often proper names;
- nulls with no plaintext value, sometimes also serving structural functions;
- fixed- and variable-length cipher elements;
- deterministic and non-deterministic segmentation [2].

Thus the v0.2 inclusion of mixed units, unequal homophone classes, nulls, and alternative key structures is historically defensible. The weak point is not the existence of those mechanisms; it is whether the exact bounded implementation adequately represents how they combine.

### Consequence for the Voynich programme

The class should not be described as an invented rescue mechanism. It is a constrained computational proxy for a documented historical family. But the proxy must be labelled precisely:

- one surface token maps to one latent plaintext unit;
- the latent unit inventory is fixed and small;
- surface legality is conditioned by the frozen PGCS cell inventory;
- global and Currier-specific keying are the only key-partition options;
- the production selector is fixed and cannot create new forms.

These restrictions make the class testable, but they also mean a negative result applies only to that proxy.

## 2. What established decipherment systems actually do

### 2.1 Heuristic search is standard, but no single objective is canonical

Dhavare, Low and Stamp use a nested hill-climb attack and explicitly evaluate success as a function of ciphertext alphabet size and ciphertext length [3]. Ravi and Knight combine letter n-gram language models and word dictionaries in a Bayesian inference framework [4]. Nuhn, Schamper and Ney use beam search with higher-order character language models [5], then show that improved rest-cost estimation and an optimized order for assigning cipher symbols can reduce a previously enormous search to a small beam [6].

This matters because v0.2 currently treats generic simulated annealing over cell assignments as if it were a sufficient test of recoverability. The literature shows that search design, assignment order, rest-cost approximation, language-model order, and lexical constraints can change the solvability of the same cipher by orders of magnitude.

### 2.2 Historical language models matter

Megyesi et al. report that historical and century-specific language models improve homophonic decipherment, particularly for older and longer ciphertexts [7]. The v0.2 external model is a deliberately small latent-unit Markov model. That is useful for controlled accounting but substantially weaker than the 5- or 6-gram character models, word constraints, and neural language models used in successful systems.

The CPU pattern is consistent with this limitation: iid-uniform homophones are recoverable more often, while frequency-weighted and sticky policies—both of which suppress simple frequency evidence—collapse recovery. A low-order plaintext model has little remaining leverage once surface frequencies and local persistence are deliberately distorted.

### 2.3 Mixed units require an explicit decoding structure

Chu, Valenti and Knight solve word-based dictionary codes by constructing a decoding lattice and searching it with a neural language model, achieving 75.1% correct cipher-word tokens on historical correspondence [8]. Aldarrab and May similarly use finite-state lattices when segmentation is ambiguous and report character-level translation edit rate; for unknown keys they separate segmentation from subsequent cryptanalysis [2].

V0.2 does not have ciphertext-boundary ambiguity because the PGCS tokenization is fixed. It nevertheless has a related latent-output problem: letters, syllables and words occupy the same one-token channel. A sequence of recovered units therefore needs a principled composition/segmentation model before it can be scored as plaintext. Treating all twelve units as states in one low-order transition matrix is not equivalent to the lattice treatment used in mixed codebooks.

### 2.4 Synthetic neural training is now a genuine benchmark family

Kambhatla, Born and Sarkar generate large synthetic datasets with random homophonic keys and train a recurrence-based Transformer to predict plaintext and recover key relations [9]. Bruton, Beloucif and Megyesi train attention-augmented LSTMs on synthetic historical English and Swedish ciphertexts with variable-length codes and simulated transcription errors [10].

The 2026 LSTM result is not a general unsupervised decipherer: it assumes a stable dataset-wide shared homophone pool and aligned ciphertext/plaintext training pairs. The authors explicitly distinguish this from independent per-document key recovery. It is still relevant as an upper benchmark and as evidence that neural models can test whether an unknown text belongs to a known key space.

### Consequence

A defensible class-level calibration should benchmark several preregistered decoder families. Failure of one annealer is a decoder failure. Failure of a frozen decoder tournament or ensemble across synthetic policies is much stronger evidence against recoverability.

## 3. Synthetic calibration and evaluation

Published work commonly uses generated ciphers with random keys, known plaintexts, and controlled variations in length, alphabet size, cipher type, variable-length codes, and noise [2,3,9,10]. Lehofer also emphasises the role of ciphertext length and the unicity point in homophonic recovery [11].

The following parts of v0.2 are therefore well supported:

- new random key for each planted trial;
- disjoint plaintext documents;
- direct key/mapping recovery metrics;
- stratification by null count, class size, key structure, and selection policy;
- production-only negatives;
- held-out predictive scoring.

The review identifies four deficiencies.

### 3.1 No length curve

The current trials use long synthetic documents. The Voynich manuscript as a whole is long, but folios, quires, sections and putative key regimes are much shorter. Recovery should be measured at multiple ciphertext lengths, not only at the full synthetic scale. The formal gate should establish the length at which each class becomes recoverable and compare that with the effective sample size under global and Currier-specific keys.

### 3.2 Selection policy is treated as nuisance rather than an inferred mechanism

The programme correctly recognised that homophone selection can make the latent key non-identifiable. The current solution—demanding robustness across iid, cyclic, frequency-weighted and sticky policies—is scientifically conservative, but the decoder does not explicitly infer or marginalise over those policies. It therefore asks one misspecified likelihood to solve four different encoders.

A future decoder should include the policy family in the model:

- fit or marginalise the policy on training data;
- charge the policy index and fitted counts;
- freeze it before held-out scoring;
- retain leave-one-policy-family-out evaluation as a separate robustness test.

No verdict should be rescued by adding a new policy after manuscript inspection.

### 3.3 Recovery metrics need output-level evaluation

The current mapping accuracy and null F1 are necessary but insufficient. Literature-standard output measures include character-level edit rate or F1, token accuracy, and segmentation error where applicable [2,10]. A future gate should report:

- cell-to-unit key accuracy, with any label symmetries handled explicitly;
- null precision/recall/F1;
- latent-unit sequence error;
- reconstructed character-level translation edit rate;
- word/subword accuracy where the output grammar permits it;
- posterior entropy or beam margin;
- key agreement across folds;
- plaintext-register identification.

### 3.4 Production controls are unusually strong and should remain

Most decipherment papers compare recovered output with known plaintext, or compare one cipher solver with another. They generally do not construct morphology- and context-matched production-only texts and ask whether the decoder hallucinates a payload. V0.2's context-iid, cell-Markov, copy-mutate and permuted-cipher controls directly address the Voynich-specific alternative that language-like surface structure is produced without encoded plaintext.

This is a substantive methodological contribution. The control suite should be expanded, not removed.

## 4. MDL and universal coding

Barron, Rissanen and Yu establish the general basis for comparing model classes through universal coding, including mixture and predictive codes [12]. This supports the programme's core requirement that a cipher explanation pay for its key, state structure, external-model choice, and data, rather than receiving a free fitted table.

The literature supports the following parts of v0.1/v0.2:

- model plus data must be charged;
- categorical supports must not be pruned after observing the data without a structure cost;
- latent paths must be marginalised or explicitly encoded;
- predictive/universal codes are preferable to uncharged maximum likelihood;
- model selection must be fixed before the target comparison.

Two distinctions are important.

### 4.1 KT/universal reporting is more theoretically canonical than the historical serializer

The fixed-support Krichevsky–Trofimov row costs and enumerative structure charges are close to standard universal-coding practice. The H-full canonical serializer is a valid two-part code only relative to its chosen description language. It is intentionally convention-dependent.

Therefore:

- the universal/KT report should be treated as the primary theoretical accounting convention;
- H-full and H-conditional should remain sensitivity analyses and operational lower/upper views;
- unanimity across conventions is a conservative programme rule, not a theorem from the MDL literature;
- a sign reversal remains a legitimate `UNRESOLVED_TABLE_COST` outcome.

### 4.2 Fitted count tables and historical key material should be separated

A historical key sheet would contain assignments, code groups, nulls, instructions, and perhaps integer multiplicities—not empirical training counts for every contextual row. When H serializes fitted counts and then also encodes the training data, it is a deliberately heavy two-part code, not a literal historical artefact cost.

Future reporting should distinguish:

- `H-key`: operational key/instructions only;
- `H-fitted`: full fitted parameter/count record;
- `I-universal`: enumerative structure plus universal data costs;
- held-out predictive cost.

No retrospective choice among them is allowed.

## 5. Interpretation of the completed v0.2 CPU run

Observed development result:

- oracle accounting/prediction: 96/96 accounting sign agreement; 92/96 predictive non-inferiority;
- learned decoder: 32/96 planted successes;
- production controls: 0/64 false positives;
- median mapping accuracy: 0.573;
- median null F1: 1.0;
- selector recovery: 1.0;
- structural recovery: 0.510;
- especially weak policies: frequency-weighted 1/24; sticky-line-reset 1/24.

The literature-informed interpretation is:

1. The class is representable and distinguishable when the key is known.
2. The current decoder is conservative and does not invent payloads in the tested controls.
3. The current inference stack cannot reliably recover keys once homophone choice suppresses simple frequency and local-transition cues.
4. The result fails the v0.2 gate exactly as preregistered.
5. It does not justify a Voynich run.
6. It does not close the bounded historical class because only one decoder family and one low-order plaintext model have been tested.

The 8xL40S rewrite is an optimizer sensitivity run. It can show whether broader batched search improves this decoder. It cannot substitute for the literature-indicated decoder and language-model comparisons.

## 6. Non-retroactive v0.3 design implied by the literature

### Stage A: preserve v0.2

Freeze the CPU result, source hash, generator, controls, seeds, and failure register. Do not tune v0.2 thresholds or reinterpret a failed criterion.

### Stage B: decoder tournament on synthetic data only

Preregister at least four decoder families:

1. nested hill climb / fixed-temperature simulated annealing;
2. beam search with optimized cipher-symbol order and admissible or calibrated rest cost;
3. Bayesian sampling combining character n-grams and lexical/subword evidence;
4. neural recurrence or sequence model trained on synthetic keys, with a strict out-of-key-space negative test.

For mixed units, add a finite-state or lattice layer that composes letters, syllables and words into candidate plaintext streams. Decoder selection must occur entirely on development generators. The selected decoder or fixed ensemble is then frozen for formal calibration.

### Stage C: external language-model registry

Use a preregistered registry containing:

- character 5- and 6-gram historical models;
- word/subword models;
- a neural historical model sensitivity;
- matched modern-language controls;
- deliberately wrong-language controls.

For eventual manuscript work, language candidates and corpora must be frozen externally; no Voynich-derived vocabulary extension is permitted.

### Stage D: policy-aware inference

Treat homophone-selection policy as a charged candidate or latent mixture. Evaluate:

- within-policy recovery;
- policy identification;
- leave-one-policy-family-out transfer;
- worst-policy recovery;
- policy-matched production false positives.

A policy that is not recoverable or distinguishable remains an explicit unresolved branch, not an opportunity for post hoc tuning.

### Stage E: expanded formal calibration

Run curves over:

- ciphertext length;
- surface alphabet size;
- number and size distribution of homophone classes;
- null rate;
- mixed-unit proportion;
- global versus partitioned keys;
- transcription noise;
- held-out surface types;
- held-out plaintext documents;
- all four selection policies.

Retain at least 80 controls per production family if the gate requires a 90% upper false-positive bound below 10% with zero observed false positives.

### Stage F: decision rule

A manuscript run becomes admissible only if the frozen formal decoder or ensemble:

- passes overall and every preregistered stratum;
- controls false positives in every production family;
- recovers output text, not only a key partition;
- transfers to unseen documents and surface types;
- remains positive under the accounting envelope;
- shows no policy-driven verdict reversal;
- passes independent reproduction and hostile model audit.

## 7. Bottom line

The literature validates the decision to demand recoverable plaintext and to calibrate on planted ciphers. It also validates the historical plausibility of mixed-unit nomenclators. It does not validate treating one simulated-annealing decoder as a class-level test.

The correct register after the CPU run is:

`V0.2 DECODER FAILED SYNTHETIC RECOVERY; BOUNDED HISTORICAL CLASS REMAINS OPEN PENDING LITERATURE-ALIGNED DECODER CALIBRATION.`

No Voynich inference follows from this review or from the v0.2 development run.

## References

[1] [Keys with nomenclatures in the early modern Europe](https://consensus.app/papers/keys-with-nomenclatures-in-the-early-modern-europe-megyesi-tudor/718f17b4eac85791a8e4765ca4307819/?utm_source=chatgpt). Beáta Megyesi, Crina Tudor, Benedek Láng, Anna Lehofer, Nils Kopal, Karl de Leeuw, Michelle Waldispühl. 2022. *Cryptologia* 48, 97–139. Consensus citation count at review date: 7.

[2] [Segmenting Numerical Substitution Ciphers](https://consensus.app/papers/segmenting-numerical-substitution-ciphers-aldarrab-may/fa8b97fa8ef553c28a1ffdda6324d5a9/?utm_source=chatgpt). Nada Aldarrab, Jonathan May. 2022, 706–714. Consensus citation count: 1. ArXiv DOI: 10.48550/arXiv.2205.12527.

[3] [Efficient Cryptanalysis of Homophonic Substitution Ciphers](https://consensus.app/papers/efficient-cryptanalysis-of-homophonic-substitution-dhavare-low/e35906b4c8a05e3dab0e0c194f57ad80/?utm_source=chatgpt). Amrapali Dhavare, R. Low, M. Stamp. 2013. *Cryptologia* 37, 250–281. Consensus citation count: 27.

[4] [Bayesian Inference for Zodiac and Other Homophonic Ciphers](https://consensus.app/papers/bayesian-inference-for-zodiac-and-other-homophonic-ravi-knight/68b9e950831151a7aac4bece7b1072be/?utm_source=chatgpt). Sujith Ravi, Kevin Knight. 2011, 239–247. Consensus citation count: 36.

[5] [Beam Search for Solving Substitution Ciphers](https://consensus.app/papers/beam-search-for-solving-substitution-ciphers-nuhn-schamper/ee4350a5c609536fa6e2ab4a93cdfd8e/?utm_source=chatgpt). Malte Nuhn, Julian Schamper, H. Ney. 2013, 1568–1576. Consensus citation count: 37.

[6] [Improved Decipherment of Homophonic Ciphers](https://consensus.app/papers/improved-decipherment-of-homophonic-ciphers-nuhn-schamper/2778240c99425c749098b9a1fa75b875/?utm_source=chatgpt). Malte Nuhn, Julian Schamper, H. Ney. 2014, 1764–1768. Consensus citation count: 15.

[7] [Historical Language Models in Cryptanalysis: Case Studies on English and German](https://consensus.app/papers/historical-language-models-in-cryptanalysis-case-studies-megyesi-sikora/cf183e8991295edd833c62889feb9187/?utm_source=chatgpt). Beáta Megyesi, Justyna Sikora, Filip Fornmark, Michelle Waldispühl, Nils Kopal, Vasily Mikhalev. 2023, 120–129. Consensus citation count: 1.

[8] [Solving Historical Dictionary Codes with a Neural Language Model](https://consensus.app/papers/solving-historical-dictionary-codes-with-a-neural-chu-valenti/03f244e9abb254e8a6f4f94862642641/?utm_source=chatgpt). Christopher Chu, Raphael Valenti, Kevin Knight. 2020. *ArXiv* abs/2010.04746. Consensus citation count: 0. ArXiv DOI: 10.48550/arXiv.2010.04746.

[9] [Decipherment as Regression: Solving Historical Substitution Ciphers by Learning Symbol Recurrence Relations](https://consensus.app/papers/decipherment-as-regression-solving-historical-kambhatla-born/f1e8606950265cf78376c4eb32e495fd/?utm_source=chatgpt). Nishant Kambhatla, Logan Born, Anoop Sarkar. 2023, 2091–2107. Consensus citation count: 4.

[10] [Attention-Augmented LSTMs for Automatic Homophonic Ciphertext Decipherment](https://consensus.app/papers/attentionaugmented-lstms-for-automatic-homophonic-bruton-beloucif/e6eff0fab204561d9255c602be93b482/?utm_source=chatgpt). Micaella Bruton, Meriem Beloucif, Beáta Megyesi. 2026. Consensus citation count: 0. ArXiv: 2606.05078.

[11] [Applying hierarchical clustering to homophonic substitution ciphers using historical corpora](https://consensus.app/papers/applying-hierarchical-clustering-to-homophonic-lehofer/d6f98a87c81552ef89c2532e951385aa/?utm_source=chatgpt). Anna Lehofer. 2021. *Cryptologia* 46, 422–438. Consensus citation count: 1.

[12] [The Minimum Description Length Principle in Coding and Modeling](https://consensus.app/papers/the-minimum-description-length-principle-in-coding-and-barron-rissanen/6e51cea380f151c4b5d93e436367a740/?utm_source=chatgpt). A. Barron, J. Rissanen, Bin Yu. 1998. *IEEE Transactions on Information Theory* 44, 2743–2760. Consensus citation count: 1,164.
