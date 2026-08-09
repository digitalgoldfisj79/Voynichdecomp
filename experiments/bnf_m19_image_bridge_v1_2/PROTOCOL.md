# BnF M19 Image-Derived Bridge v1.2 — Frozen Protocol

Date: 2026-08-09
Seed namespace: `M19IMAGEv12`
Parent: `experiments/bnf_m19_why_german_v1_1/RESULT.md`
Image source: private HF dataset `Digitalgoldfish79/vdino3-crops` at revision `ea597db8ff2c06631c4c311d90c8cf0418f5e26c`.

## Question

Does the BnF Latin 7342 19-value unmarked numerical channel (M19) produce a language-specific, out-of-sample signal when the Voynich surface representation is derived from manuscript images rather than EVA ASCII characters?

This is a representation-falsification experiment. The v1.0 literal-EVA German result is not treated as prior evidence for any language.

## Sealed fields

The runner MUST NOT read the following manifest fields before the terminal verdict: `word`, `eva_aligned`, `eva_glyph`, `word_len`, or any other transliteration label/string. They may be inspected only in a post-verdict audit.

Permitted image/provenance fields are `id`, `folio`, `kind`, `view`, `word_index`, `slot`, `n_slots`, `low_conf` and DINO vectors. Word boxes are inherited image regions from the recovered pipeline; no textual label inside the box is used.

Primary image stream: `kind=ccmerge`, `view=norm`, `low_conf=false`. DINO source: `results/corpus_embeddings_full.npz`, 768-dimensional normalized CLS vectors. Dense DINO is reserved as a replication only if a primary signal survives C12.

## Frozen folio split

The 225 `ccmerge/norm` folios are sorted by SHA-256 of `M19IMAGEv12split::<folio>` and partitioned 50/20/30:

- T12: first 112 folios — image/cluster fitting and all model selection.
- H12: next 45 — first Voynich language test. No language-dependent fitting choice may be made after H12 is scored.
- C12: final 68 — sealed confirmation. C12 embeddings/sequences may not be language-scored until H12 has passed the relevant arm's primary gate.

T12 is internally split 80/20 by SHA-256 of `M19IMAGEv12vis::<folio>` into Tfit/Tvis for image-model selection. H12 is not used to choose representation, K, centroids, segmentation parameters, or solver budget.

The exact T12/H12/C12 folio lists are fixed by this algorithm and were printed in the preflight before any DINO/M19 language score.

## BnF channel

Use the exact five BnF tables frozen in v0.7–v1.1. The unmarked channel contains 19 numerical values. A surface class maps to exactly one numerical value. For K surface classes, legal maps are surjective onto all 19 values and each value receives one or two surface classes; therefore K is restricted to {19,25,31,38} (0, 6, 12 or 19 duplicated numerical values respectively).

The hidden plaintext emission law remains the v0.9 law: for a plaintext letter, choose uniformly among its distinct BnF values. No table identity is transmitted.

## Arm A — strict image-component surface classes

1. Use only ccmerge/norm DINO vectors and image ordering `(folio, word_index, slot)`.
2. Test two image representations, chosen without language information:
   - R0: raw unit-normalized DINO CLS vector.
   - R1: folio-centred residual `normalize(x - mean_folio(x))`.
3. For each R in {R0,R1} and K in {19,25,31,38}, fit MiniBatchKMeans on Tfit only, fixed seeds 408 and 409.
4. Match seed-409 centroids to seed-408 by Hungarian maximum cosine. On Tvis compute:
   - assignment stability after centroid matching;
   - cosine silhouette on a deterministic <=10,000-vector Tvis sample;
   - cluster recurrence across Tvis folios;
   - 5th-percentile Tfit cluster-similarity thresholds and accepted Tvis fraction.
5. Image gate for a candidate representation/K: stability >=0.75; accepted Tvis fraction >=0.75; every cluster occurs in >=3 Tvis folios and has >=25 Tvis assignments. Among passing candidates choose maximum Tvis cosine silhouette; ties within 0.005 choose smaller K, then R1 over R0. If none pass, Arm A is IMAGE-UNDERPOWERED and Arm B may still proceed from the highest-silhouette candidate, but Arm B must independently pass its own visual gate.
6. Refit the chosen K/representation on all T12. Freeze centroids and 5th-percentile per-cluster similarity thresholds. H12/C12 assignments below their centroid threshold are unmapped breaks; coverage must be reported.

### Arm A positive controls

Before H12 language scoring, qualify the symbolic M19 solver at the selected K using fresh 45,000-train / 39,000-hold synthetic controls for Latin, Italian, German, French, Arabic and Spanish, generated from the exact M19 law with a deterministic legal K->19-value map and opaque surface labels.

Gate: 6/6 correct language rank; minimum correct-language margin >=0.05 nats/letter; median numerical-map recovery >=0.95; minimum recovery >=0.85; minimum independent-fit agreement >=0.90. If the gate fails, no H12 score is admissible.

### Arm A Voynich primary gate

Fit a separate legal cluster->M19 map under each of the eight frozen language models (Latin, Italian, German, French, Ancient Greek, Hebrew, Arabic, Spanish) using T12 only. Two independent fits per language.

Evaluate fixed fitted maps on H12 by exact M19 forward likelihood with word boundaries preserved.

Primary H12 signal requires all of:
- top language rank 1;
- margin to runner-up >=0.05 nats/mapped unit;
- independent-fit agreement of top language >=0.90 occurrence-weighted;
- H12 mapped-unit coverage >=0.90.

If H12 fails, do not score C12 under Arm A.

If H12 passes, freeze the winning T12 map verbatim and score C12 without refitting. C12 confirmation requires candidate rank 1, margin >=0.05, coverage >=0.90, and candidate-minus-runner-up margin >0 in all four deterministic C12 folio buckets (`sha256(M19IMAGEv12bucket::<folio>)[0] mod 4`).

## Arm B — visual-only segmental surface classes

Arm B is attempted if Arm A does not produce a confirmed C12 signal or if the strict component stream is visibly oversegmented. It is not allowed to use any plaintext/language score to choose segmentation.

Starting from the best Tfit/Tvis Arm-A image representation and K:

1. Adjacent ccmerge components within each word may form a segment of length 1, 2 or 3.
2. Segment embedding is the unit-normalized mean of member DINO vectors.
3. Fit centroids and segment words by alternating three times:
   - centroid fit on Tfit segment embeddings;
   - dynamic-programming segmentation minimizing `sum(1 - max_cosine(segment, centroid) + lambda)`.
4. Scan lambda in {0.02,0.04,0.06,0.08,0.10,0.12} using Tfit/Tvis only. Choose the lambda with maximum Tvis cosine silhouette subject to assignment stability >=0.75, Tvis accepted coverage >=0.75, every cluster recurring in >=3 Tvis folios, and mean segments/word between 2.0 and 10.0. Ties within 0.005 choose larger lambda (simpler/fewer segments).
5. Refit/freeze segmentation and centroids on all T12. H12/C12 are segmented with frozen centroids/lambda and no language feedback.

Arm B must pass the same six-language positive-control gate and the same H12/C12 language gates as Arm A. Positive controls for Arm B additionally split each synthetic surface symbol into 1–3 microcomponents under a frozen geometric distribution, generate microcomponent embeddings around fixed class prototypes, and require the segmental instrument to recover the correct language and >=0.85 symbol-boundary F1 before Voynich is admissible.

## Confound/null tests after any C12 confirmation

A confirmed candidate must survive:

1. **Order null:** shuffle image units within each word on C12, 200 deterministic replicates. The observed candidate margin must exceed the 99th percentile of the null.
2. **Frequency null:** independently permute cluster labels within each folio while preserving cluster counts and word lengths, 200 replicates. Same 99th-percentile criterion.
3. **Boundary null:** concatenate words within folio, removing normal word starts/ends. Report whether candidate remains rank 1; this is diagnostic, not a hard gate.
4. **Dense-DINO replication:** repeat fixed image partition and fixed numerical map with `corpus_embeddings_full_dense.npz`, assigning by dense centroids learned on T12 only. Candidate must remain rank 1 on C12; margin >=0.03.
5. Only after all primary tests, unseal EVA audit fields and report NMI/agreement between image classes and EVA labels. This audit cannot change the verdict.

## Verdict vocabulary

- `IMAGE INSTRUMENT NOT QUALIFIED`
- `NO IMAGE-M19 SIGNAL`
- `H12 IMAGE-M19 CANDIDATE / C12 FAILED`
- `CONFIRMED IMAGE-M19 SIGNAL <language>`
- `CONFIRMED BUT NULL-SENSITIVE`

No decoded strings may be inspected before a C12 confirmation. A statistical signal is not a plaintext claim without readable, independently constrained recovery.
