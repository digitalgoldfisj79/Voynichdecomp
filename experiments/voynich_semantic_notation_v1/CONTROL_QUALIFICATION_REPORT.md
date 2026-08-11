# VSN-v1 Synthetic / Control Qualification Report — Stage 1

Date: 2026-08-12
Status: DISCOVERY qualification halted before CONFIRMATION because the frozen power gate fails.

## Controls completed

### 1. Cardinality / source-integrity control — PASS

An initial occurrence-table join duplicated foldout metadata and inflated 36,680 RF token occurrences to 36,888 rows. The construction was rejected, duplicate canonical-folio metadata were collapsed before joining, and the final snapshot exactly matches the direct RF split at 36,680 occurrences. No target statistic from the failed table was retained.

### 2. Outcome-blind morphology construction — PASS

7,893 RF token types produced 28,435 exact edit-distance-1 pairs and 361 support-qualified prefix/suffix candidates without any visual-target join. Commonly discussed forms such as `qo-`, `-dy`, `-ey`, `-edy` and `-aiin` arise automatically.

### 3. Discovery / confirmation firewall — PASS

Block assignment is deterministic and outcome-blind. Approximately one quarter of quire/folio blocks are sealed CONFIRMATION. All target statistics in Stage 1 use DISCOVERY only. Confirmation image embeddings have not been queried.

### 4. Cross-family transfer — PASS for `-dy` / whole plant in DISCOVERY

Training and test residual token cores are disjoint. `-dy` whole-plant TEST-core mean cosine alignment is 0.18544 against the TRAIN-core visual direction, with 38 TRAIN and 36 TEST cores.

### 5. Discovery multiple-testing / sign-flip null — PASS for `-dy` / whole plant

1,024 deterministic sign flips over TEST-core alignments:

- observed 0.18544;
- null p99 0.12590;
- empirical p 0.000976;
- BH q 0.01795.

### 6. Leave-one-quire-out — PASS for `-dy` / whole plant

Seven held-out quires are all positive. Mean LOQO alignment = 0.091921. Quire means range from 0.027974 to 0.203731.

### 7. Block bootstrap — PASS for `-dy` / whole plant

Fixed seed 20260812; 200,000 bootstrap resamples of seven quire means:

- bootstrap mean ≈ 0.09192;
- 95% percentile interval ≈ [0.05125, 0.13813].

The interval excludes zero.

### 8. Quire-dominance ceiling — PASS

Using TEST-core-count weighted positive contributions, the largest quire contributes about 37.5% of total positive gain, below the frozen 50% ceiling.

### 9. Within-quire visual page-reassignment null — PASS for `-dy` / whole plant

A 3,072-D all-permutation implementation first failed transactionally due Supabase temporary-disk exhaustion. A mathematically equivalent scalar implementation projected pages onto the frozen TRAIN-core direction before permutation. This changes computation, not the tested direction or null hypothesis.

256 deterministic bijective page reassignments within quire:

- observed effect 0.00574543;
- null mean 0.00008702;
- null SD 0.00156223;
- null p99 0.00306116;
- 0/256 permutations >= observed;
- corrected empirical p = 1/257 = 0.003891.

### 10. Frequency/family confounding — PARTIALLY CONTROLLED

The primary comparison is within residual token family/core, and train/test cores are disjoint. This strongly reduces full-token frequency and lexical-family confounding. Explicit nested-affix deconvolution and a dedicated frequency-only predictive BASE have not yet been completed.

## Controls not completed

### Structured slot/Markov pseudo-text null — NOT RUN

The protocol requested a generated structured-text null preserving token/component frequencies, morphology, section and line-position biases while breaking image linkage. The exact within-quire page reassignment already preserves the *actual* text structure rather than an approximation and breaks the text-image linkage, so it is a stronger linkage randomization in one sense. Nevertheless it is not the explicitly requested slot/Markov generator and is recorded as unrun, not silently substituted.

### Spatial word-object null — PARTIAL ONLY

The populated database contains no token-level x/y word table. The page-reassignment test is a page-level linkage null; it cannot test token-to-object distance because those coordinates are unavailable.

### Hand control — UNAVAILABLE

The populated catalogue fields for Davis/Currier hand are null for the RF-linked records. They were not imputed.

### STA / AAA replication — NOT RUN

Per protocol, alternate-representation replication comes only after RF discovery qualification. It cannot repair confirmation power and was therefore not used to justify opening the sealed arm.

## Decisive gate: confirmation power — FAIL

Support-only inspection of the sealed herbal CONFIRMATION arm shows:

- 2 confirmation quire/folio blocks containing plant text/targets;
- 17 plant folios;
- 9 matched `-dy` residual cores / core×quire strata;
- 37 status pages.

Discovery TEST-core standardized effect is approximately 0.713. Applying the frozen 25% shrinkage gives planning d≈0.534.

Approximate one-sided power:

- n=9 matched cores: ~43%;
- n=17 folios: ~68%.

Both are below the preregistered >=80% power gate. Treating the 37 status pages as independent would violate the clustering principle.

**QUALIFICATION DECISION: FAIL AT POWER GATE. CONFIRMATION MUST REMAIN SEALED.**

This is not a negative result for the `-dy` discovery effect. It is a negative result for the ability of the frozen VSN-v1 herbal split to confirm it formally.

## Compute discipline

No Hugging Face jobs were launched for Stage 1. A final HF job-status check on 2026-08-12 returned **no running jobs**.
