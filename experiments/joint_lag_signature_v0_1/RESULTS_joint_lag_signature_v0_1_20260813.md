# Joint lag-signature programme v0.1 — results

Seed: 20260813. Tight null: **within-line multiset-preserving permutation**. Reference ZLZI n=37,465 tokens, 5,162 lines; 2,000 permutations.

## Controls

| control/stat | result |
|---|---|
| C0 E2 | mean ratio 1.007; |z|>=2 in 0/20; PASS |
| C0 N1 | mean ratio 0.999; |z|>=2 in 0/20; PASS |
| C0 BRIDGE | mean ratio 1.008; |z|>=2 in 2/20; PASS |
| C0 CHAIN5 | mean ratio 0.881; |z|>=2 in 0/20; **FAIL** |
| C1_ABA | PASS |
| C2_BRIDGE | PASS |
| C3_PARITY | PASS |

Valid downstream statistics: `BRIDGE=true, CHAIN5=false, E2=true, E4=true, ISO_ABA=true, N1=true, PHASE=true, TOPK=true`.

## Primary decisions

| hypothesis | verdict |
|---|---|
| H0_N1 | **PASS** |
| H0_E2 | **PASS** |
| H1_BRIDGE | **UNRESOLVED** |
| H2_PARITY_CHAIN | **UNRESOLVED** (CHAIN5 control failed) |
| H3_ISOLATED_CLOSURE | **NOT SUPPORTED by formal tree** |
| H4_LINE_PHASE | **SUPPORT** |
| H5_LEXICAL_CARRIER | **TECHNICAL PASS, POWER-CONFOUNDED; DO NOT INTERPRET AS ESTABLISHED** |

## Reference-frame metrics

| metric | actual | null | ratio | z |
|---|---:|---:|---:|---:|
| E1 | 301 | 299.754 | 1.004 | 0.083 |
| E2 | 319 | 262.914 | **1.213** | **4.012** |
| E4 | 203 | 188.322 | 1.078 | 1.179 |
| N1 | 1245 | 1062.101 | **1.172** | **6.752** |
| BRIDGE | 21 | 20.868 | 1.006 | 0.030 |
| CHAIN5 | 4 | 6.120 | 0.654 | -0.866 — **invalid downstream statistic** |
| ISO_ABA | 227 | 176.643 | **1.285** | **4.192** |

## Cross-frame identical-statistic replication

| frame | tokens | E2 ratio (z) | N1 ratio (z) | BRIDGE ratio (z) |
|---|---:|---:|---:|---:|
| GCGA | 39,535 | 1.200 (3.284) | 1.123 (5.802) | 1.132 (0.664) |
| VDRB-1 | 37,899 | 1.243 (4.273) | 1.136 (5.411) | 0.996 (-0.020) |
| TTVE | 37,886 | 1.194 (3.547) | 1.148 (5.700) | 1.027 (0.118) |
| TTIA | 37,759 | 1.196 (3.479) | 1.145 (5.668) | 1.037 (0.176) |
| ZLZB | 37,479 | 1.213 (3.963) | 1.172 (6.545) | 1.000 (0.001) |
| ZLZI | 37,465 | 1.214 (4.027) | 1.174 (7.298) | 1.008 (0.037) |
| TTLI | 34,351 | 1.162 (2.995) | 1.163 (6.510) | 1.201 (0.952) |
| VDRB | 34,038 | 1.169 (2.812) | 1.148 (5.864) | 0.747 (-1.058) |
| FFSG | 33,095 | 1.289 (5.146) | 1.099 (4.175) | 1.244 (1.086) |
| FFSG-2 | 32,421 | 1.293 (5.007) | 1.096 (3.789) | 1.372 (1.445) |
| RGVN | 20,059 | 1.188 (2.401) | 1.146 (4.891) | 1.336 (1.239) |
| PCCA | 16,110 | 1.183 (2.326) | 1.137 (4.376) | 1.167 (0.599) |

H0-N1 passes: all 12 unique transcription contents have N1 ratio >1; 10/12 meet ratio>=1.10 and z>=2; median ratio 1.145. H0-E2 passes: 12/12 meet ratio>=1.10 and z>=2; median ratio 1.198.

## Coupling test: H1

`BRIDGE = A-B-A` with `A==A` at lag 2 and literal-frame Levenshtein(A,B)=1. On ZLZI it is exactly chance: ratio **1.006**, z **0.030**. No cross-frame BRIDGE result reaches z=2. The formal preregistered verdict is UNRESOLVED only because its falsification criterion also required cross-frame median <=1.05; observed median is ~1.085. There is therefore **no positive evidence that the N1 and E2 effects are concentrated in the same A-near-B-A triples**.

## Persistence versus isolated closure

- E2: ratio 1.213, z 4.012.
- E4: ratio **1.078**, z **1.179** — no evidence of lag-4 continuation.
- CHAIN5 failed calibration and is not interpreted.
- ISO_ABA: ratio **1.285**, z **4.192**.

The formal H3 tree cannot pass because H2 is unresolved after CHAIN5 failed its control. Descriptively, however, the calibrated statistics favour a strong isolated lag-2 closure over persistent parity continuation: E2 and ISO_ABA resolve strongly while E4 does not.

## H4: line phase / boundary

- even-start E2: ratio 1.105, z 1.358.
- odd-start E2: ratio 1.339, z 4.014.
- **line-start i=0 E2: ratio 0.611, z -2.499.**
- **interior i>=1 E2: ratio 1.314, z 5.307.**
- parity contrast: z -1.924, enrichment fold-contrast 1.211 — does not pass.
- **boundary contrast: z -4.146, enrichment fold-contrast 2.149 — passes.**

Thus H4 support is specifically a **line-boundary effect, not an odd/even parity effect**: first-to-third word identity is suppressed at line starts while the lag-2 exact surplus is strongly interior.

Exploratory cross-frame check using the same frozen statistic: interior E2 enrichment exceeds 1.24 and z>3 in all 12 unique frames. The boundary contrast is negative in all 12 and z<-3 in 9/12. This is not an independent replication because the frames transcribe the same ink, but it shows the boundary result is not an EVA segmentation artefact.

## H5: high-frequency lexical carriers — preregistered criterion is defective

- mask top 5 repeated endpoint types: E2 ratio 1.198, z 3.230.
- mask top 20: ratio 1.150, z 1.771.
- mask top 50: ratio 1.168, z 1.448.

The preregistered H5 rule technically calls this SUPPORT because significance drops below z=2 after masking top 20. That decision rule confounds **loss of power** with **loss of effect**: the effect-size ratio remains ~1.15–1.17 after masking 20–50 types. Therefore H5 is recorded as a technical preregistered pass but **not interpreted as evidence for a small lexical carrier**. A corrected v0.2 test must compare top-K enrichment against rest directly under the same permutations.

## Execution / recovery note

An orchestration-level duplicate local launch survived an initial transport timeout and later cleared the first output directory. The completed first run's decisions and controls had already been captured. The detailed cross-frame/reference artefacts were reconstructed deterministically from the frozen code, seeds and data; the recovered decisions are identical to the completed first-run summary. No threshold, statistic, hypothesis or seed was changed. SHA-256 of recovered full JSON: `5d74270b887014689662662d29f8d7c833cf3d346863d9ef0968afd704b25646`.

## Interpretation discipline

- A PASS is only a pass of the preregistered statistical implication. It does not identify plaintext, cipher, hoax, or scribal intent.
- July G' sufficiency concerned the older loose/page-null profile and is not treated as explaining a residual that survives this tighter within-line null.
- Cross-frame decisions use 12 unique transcription contents; four byte-identical aliases are excluded from replication counts by preregistered amendment.
