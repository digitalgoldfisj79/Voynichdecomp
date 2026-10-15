# VBM v141: identifiability-first admission review
14 October 2026  |  Status: **NO-GO for the unmodified v138/v140 architecture**

## RETRACTED FINDINGS AND PROPOSALS (kept at the top)

1. **Retracted as a viable immediate engineering plan:** “Keep the existing four tables and simply install strict assertions plus a decoder to make v141 lossless.” That is impossible with the existing one-cell-per-cluster inventory on the supplied source: 722 distinct types compete for only 443 cells, while the frozen rank assignment also conflicts with the vowel-bridge table on 13,369 of 85,364 checked transitions. A revised architecture needs a new, explicitly justified codebook/segmentation rule; stricter assertions alone cannot produce a working cipher.
2. **Retraction of evidential interpretation, not of reproduction:** The source author's v138 resemblance/AUC and prior v140 homophony improvements are not demonstrations of mechanism or recoverability. Previous independent audit showed the same AUC acceptance threshold admitted strongly order-scrambled Voynich as a hard negative. **Do not use these metrics to decide whether VBM is historically responsible for Voynich.**

## Admission scope, hypothesis, and prior failures

**In scope:** Exact combinatorial feasibility of the existing v138 tables on the supplied German source, using the supplied 443 VMS-derived cells and vowel-pair table. Test whether merely reassigning cells, without new symbols or side information, could make a uniquely decodable key-compatible cluster mapping. **Out of scope:** Any claim about plaintext of Voynich, historical plausibility, its source language, or its generation mechanism. The architecture reuses the same German corpus from which v138 was fitted; it is *not* a held-out validation.

Precommitted `admission_contract.json` defines necessary criteria before running the independent allocation calculation. The source ZIP and contract have immutable SHA-256 fingerprints in `stage0_provenance.pkl`. The original author code and reference output are kept intact; the source is extracted into a quarantined working directory.

Mechanism alternatives:
- **M1:** A fixed, injective, globally decodable codebook whose vowel-bridge state predicts the next cell key.
- **M2:** The actual v138 rank code with cell collisions and state/fallback overrides. It may render Voynich-like text without preserving the entire plaintext.
- **M3:** An autonomous slot/token process producing exactly the same distribution (or even the same observed string) as M1. On Voynich text alone, M1 and M3 are observationally indistinguishable. No output-only statistical score can uniquely select M1 without an independent linguistic/source-output anchor.

M1 vs M2 *is identifiable on known synthetic/source-cipher pairs* by exact inverse decoding. M1 vs M3 *is not identifiable from ciphertext alone*. Thus the only permitted run at this stage is an **engineering feasibility preflight**, not another Voynich generative similarity test.

## Exact preflight results

| Necessary property | Verified result | Gate |
|---|---:|---|
| Source consonant-cluster types using source-author event extraction | 722 | — |
| Frozen distinct VMS-B cells available | 443 | — |
| At least one unique cell per cluster | At least 279 types cannot have a unique cell (pigeonhole bound) | **FAIL** |
| Frozen v138 rank allocation | 147 cells shared by 426 cluster types | **FAIL** |
| Frozen vowel-bridge compatibility | 13,369 / 85,364 known-next events (15.66%) have no pair to their assigned next key | **FAIL** |
| Optimally reallocated **distinct** cells with compatible incoming vowels | 443 / 722 types, covering 88,751 / 91,234 nonempty-cluster events (97.28%) | **FAIL: 279 types left unmapped** |
| Events uncovered even by the optimised injective assignment | 2,483 / 91,234 (2.72%) | **FAIL** |
| Exact symbolic inverse of the *restricted* matching subset | 443 / 443 table entries map uniquely back to their source type | **PASS for mapping algebra only** |
| Hard negative: force one same-key collision while retaining key compatibility | Detected and rejected | **PASS for mapping collision test only** |
| Independently rerun the constrained optimum | SciPy Hungarian = NetworkX min-cost-flow: 88,751 covered events | **PASS for computational replication** |
| Independent end-to-end plaintext decoder | Not implemented; prior prerequisites fail | **NOT TESTED, not a pass** |
| Within-section blind Voynich holdout / matched mechanism negatives | Not permitted at this stage | **NOT TESTED** |

All reported figures are deterministic whole-input counts. The *null standard deviation is not applicable*; there is no simulated null population or sampling-based significance claim. Comparisons between the original frozen map and the optimized reallocation are **engineering bounds**, not estimates of what a manuscript writer could historically achieve.

The maximum-assignment bound assumes a relaxed version of the problem: it enforces key-family compatibility for observed incoming vowels and a distinct cell per assigned source cluster. It does **not** require that each *complete emitted token* satisfies the observed VMS atom bigram/trigram constraints, nor does it guarantee recovery of punctuation, spelling, word boundaries, long prefix joins, or future out-of-vocabulary source clusters. Its 97.28% figure is therefore *not* decoding accuracy and not a lower bound on achievable full-cipher recovery.

## Controls and governance verification

1. **Circularity:** The reference VMS-B was used to select cells and bridge pair inventory. The German source was used to rank its clusters. Therefore all in-sample resemblance figures are contaminated by fitting and are excluded from this admission decision.
2. **Leakage:** No held-out Voynich sections are used for fitting or evaluation. No v141 candidate has been tested on Voynich. The necessary-condition counts apply only to the supplied known source and fixed inventory.
3. **Confounds:** A lossless code using an arbitrary additional escape mechanism is trivial and does not establish that *VBM* is lossless or historically plausible. Such escapes were expressly excluded from the current gate.
4. **Matched nulls / positive controls:** The exact assignment test has a positive (unique compatible matching) and a near-identical negative (a same-key collision). A more demanding cipher-mechanism test with section-matched slot and Markov negatives would be required later, but is not justified until an inverse decoder exists.
5. **Control fairness:** Both assignment controls use the same key support, cell inventory, source clusters and admissibility logic; the negative differs only by collision. The independent solver uses different mathematics to verify the upper bound.
6. **Measurement degeneracy:** Distinct cells can be counted exactly; true word-level recovery cannot be inferred from this count. The match does not test the decoding of glyph strings or original word spacing.
7. **Representation dependence:** Changing EVA/REVA, atomic segmentation, eligible cell inventory, vocabulary normalisation, or source manuscript section changes the problem. None was varied post hoc.
8. **Decision-rule fragility:** Admission required all types uniquely represented and zero incompatible transitions. No threshold was relaxed to accept a 97.28% event-coverage result.
9. **Audit completeness:** Baseline hash, frozen gates, independent recheck, both controls, and failure checkpoint are included. No full decoder, blind prediction or manuscript-inference experiment is claimed.

## Explicit next architectural decision (proposal only, not an admitted experiment)

The user/author would need to authorise an **architectural replacement**, not an incremental v138 patch, with a precommitted code specification that includes:

- At least 722 distinct, *self-delimiting* decodable cluster representations for the *observed training inventory*, with an explicit out-of-vocabulary rule for genuinely unseen clusters. Reusing the 443 cells means some clusters must use multi-unit codes or a demonstrably invertible prefix channel; 279 is only the lower bound for *additional distinguishable cluster codewords*, not necessarily 279 new glyphs.
- A single constrained assignment rule that determines the cluster representation, incoming/outgoing keys and vowel bridge *jointly*; no unlogged fallback to a different vowel, key or prefix.
- An explicit, decipherable source-word-boundary channel (or a declared measurement limited to the continuous normalised character stream); loss from `norm_word` must be counted separately.
- A frozen decoder specified and implemented independently of the encoder. Sidecar ledgers, message-specific rebuilding of rank tables and silent exceptions are prohibited.
- Precommitted engineering controls: paired artificial examples with known keys/normalisation; same-count, same-key collisions and altered-bridge matched negatives; unknown cluster and word-boundary stress tests; exact end-to-end character-stream recovery requirement of 100%, or a declared *different*, explicitly lossy hypothesis.

After (and only after) that, a **separate identifiability audit** must specify an independent plaintext/linguistic anchor or a feature that cannot be matched by an autonomous slot model with equal data access, frozen by section and checked with blind holds. Without such an anchor, a reversible cipher is a *proof of construction*, not evidence of Voynich's generating mechanism.

**Decision:** Current architecture = **NO-GO**; cease similarity optimisation and do not publish a v141 cipher or mechanism claim. A redesign could later be admitted under a new frozen specification, but is not automatically promoted from this failed one.

## Reproduce and independently verify

From the directory containing the untouched supplied ZIP:

```bash
python vbm_v141_gate/preflight.py Generator_v138_Pruefpaket_2026-10-14.zip
python vbm_v141_gate/adversarial_review.py
```

Dependencies: Python 3, NetworkX, NumPy and SciPy. Outputs are saved atomically as `stage0_provenance.pkl`, `stage1_baseline.pkl`, `stage2_optimized_bound.pkl`, `stage3_admission.pkl`, `stage4_adversarial.pkl`, and inspectable `results.json` and `independent_review.json`. These numerical statements can be recalculated against the supplied ZIP; no web-derived or speculative numbers are used.
