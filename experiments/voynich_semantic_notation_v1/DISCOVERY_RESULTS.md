# VSN-v1 Discovery Results — Herbal Annotation-Free Arm

Date: 2026-08-12
Status: **DISCOVERY ONLY. CONFIRMATION EMBEDDINGS REMAIN UNOPENED.**

## Visual data

Existing Supabase assets supplied the first annotation-free target without new GPU inference:

- 114 deterministic non-spurious whole-plant targets;
- 114 deterministic non-spurious root targets;
- existing embedding dimension: 3,072.

Only DISCOVERY blocks were used for the results below.

## Test construction

For each support-qualified component, matched component-present vs alternative-component forms were compared **within residual token family/core and within quire**. The resulting 3,072-D visual contrast vector was averaged per residual core.

Residual cores were split outcome-blind into TRAIN and TEST by deterministic SHA-256. A component direction was fitted on TRAIN cores only. TEST-core visual deltas were scored by cosine alignment with that frozen direction. This is therefore a cross-family transfer test rather than a full-token association test.

## Initial cross-family screen

Among component×visual-part tests with at least 10 TRAIN and 10 TEST cores, a 1,024-sign-flip empirical null was applied to TEST-core alignments and Benjamini-Hochberg correction was applied across the discovery family.

The strongest initial signals formed a nested suffix cluster:

| component | visual part | TRAIN cores | TEST cores | mean held-out cosine | null p99 | empirical p | BH q |
|---|---|---:|---:|---:|---:|---:|---:|
| `-edy` | plant | 21 | 13 | 0.34489 | 0.23201 | 0.000976 | 0.01795 |
| `-hdy` | plant | 12 | 14 | 0.30299 | 0.20516 | 0.000976 | 0.01795 |
| `-hdy` | root | 12 | 14 | 0.25003 | 0.18740 | 0.000976 | 0.01795 |
| `-dy` | plant | 38 | 36 | 0.18544 | 0.12590 | 0.000976 | 0.01795 |
| `-ey` | plant | 36 | 29 | 0.13747 | 0.10857 | 0.000976 | 0.01795 |
| `-edy` | root | 21 | 12 | 0.29805 | 0.21749 | 0.001951 | 0.02564 |

Several additional pairs passed BH q<=0.05 at this screen. They were **not** accepted as operator evidence because leave-quire-out was mandatory.

## Leave-one-quire-out hostile control

The TRAIN direction was refitted with each evaluation quire excluded; evaluation used TEST residual cores in the held-out quire.

Key results:

| component | part | held-out quires | mean LOQO alignment | positive quires |
|---|---|---:|---:|---:|
| `-hdy` | plant | 5 | 0.11011 | 5/5 |
| `-edy` | root | 5 | 0.10151 | 5/5 |
| `-dy` | plant | 7 | 0.09192 | **7/7** |
| `-edy` | plant | 5 | 0.08127 | 4/5 (one ≈0) |
| `cho-` | plant | 8 | 0.08005 | 5/8 |
| `ok-` | root | 7 | 0.03017 | 5/7 |
| `-ey` | plant | 8 | 0.02578 | 6/8 |
| `-dy` | root | 7 | 0.00844 | 5/7 |
| `-aiin` | root | 7 | 0.00068 | 4/7 |
| `-ey` | root | 8 | 0.00030 | 4/8 |
| `-hdy` | root | 5 | -0.01226 | 3/5 |

This removes much of the apparent discovery family. In particular, the root effects of broad `-dy`, `-ey`, `-aiin`, and `-hdy` do not survive robustly.

The broadest surviving signal is `-dy` on **whole-plant** embeddings. Its seven held-out quire means were:

- q01 0.137816
- q02 0.027974
- q04 0.060542
- q05 0.069309
- q06 0.112920
- q07 0.203731
- q08 0.031155

A fixed-seed 200,000-replicate block bootstrap over these seven quire means gives mean 0.091921 and 95% percentile interval approximately **[0.05125, 0.13813]**. The largest weighted quire contribution is approximately **37.5%**, below the frozen 50% dominance ceiling.

## Page-reassignment linkage null

A full 3,072-D × 1,024 permutation implementation was attempted first and failed transactionally because Supabase exhausted temporary disk. A scalar-equivalent implementation was then used: every page was projected onto the already frozen TRAIN-core `-dy` direction, and visual pages were bijectively reassigned **within quire**. This preserves the quire visual distribution while breaking the text-image linkage.

For `-dy` / whole plant, 256 deterministic within-quire reassignments completed:

- TEST residual cores: 36
- observed scalar projection effect: **0.00574543**
- null mean: 0.00008702
- null SD: 0.00156223
- null 99th percentile: **0.00306116**
- permutations matching/exceeding observed: 0/256
- one-sided empirical p with +1 correction: **1/257 = 0.003891**

Thus the broad `-dy` whole-plant discovery signal survives this hostile page-linkage null.

## Interpretation at this stage

This is the first VSN-v1 result that warrants further scrutiny. It says only:

> Across disjoint RF residual token families, changing the suffix slot to `dy` is associated in DISCOVERY data with a repeatable direction of change in whole-plant visual embedding space, surviving quire exclusion and within-quire visual reassignment.

It does **not** establish semantics, a meaning for `dy`, a cipher, or a spoken-language morpheme. Nested suffixes (`-edy`, `-hdy`) cannot yet be interpreted independently from the broader `-dy` effect.

## Confirmation power blocker

Before opening any confirmation embedding, only support counts were inspected in the sealed arm.

The frozen CONFIRMATION split contains, for this herbal endpoint:

- 2 confirmation blocks with plant text/visual targets;
- 17 plant folios;
- 153 `-dy`-related residual cores seen at all;
- only **9** residual cores with both component-present and alternative forms matched within quire;
- 9 matched core×quire strata / 37 status pages.

Using the discovery TEST-core standardized effect (about 0.713) and the preregistered 25% shrinkage penalty gives a planning effect around 0.534. Approximate one-sided power is only **~43% at n=9 matched cores** and **~68% even at n=17 folios**, below the frozen >=80% gate. Treating 37 status pages as independent would violate the clustering principle and is not permitted.

Therefore **confirmation is not opened** under VSN-v1 at this stage. The discovery effect remains a qualified-looking but formally unconfirmed signal.

## Controls still unrun because the power gate already fails

- dedicated structured slot/Markov pseudo-text generator null;
- nested-suffix deconvolution (`-dy` vs `-edy/-hdy` as independent effects);
- STA-family/full-STA/AAA representation replication;
- cross-domain transfer.

These may be useful diagnostics, but none can convert the existing underpowered sealed herbal arm into a valid confirmation sample. No confirmation threshold will be relaxed post hoc.
