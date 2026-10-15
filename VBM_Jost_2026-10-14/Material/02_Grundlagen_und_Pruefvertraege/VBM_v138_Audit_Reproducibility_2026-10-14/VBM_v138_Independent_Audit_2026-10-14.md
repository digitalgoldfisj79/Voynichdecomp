# Independent audit: VBM generator v138 (14 October 2026)

**Scope.** Audit of the supplied `Generator_v138_Pruefpaket_2026-10-14.zip` (SHA-256 `e67fe89555f8d8b2e74218077bb186f4f6fec4e5444f0ee9c289750d1efd4033`). Source code, reference output, datasets and configuration files were inspected. Tests were run on the supplied material. These findings concern the specific supplied implementations, not the logical possibility of a different VBM system.

## RETRACTED OR CORRECTED CLAIMS

1. **"Every output token contains only attested VMS-B atomic bigrams/trigrams." CONTRADICTED.** 2,365 of 101,297 final tokens have at least one unattested atomic bigram; 7,259 have at least one unattested atomic trigram. These counts use the package's own atomisation and VMS-B reference data.
2. **"Each cluster has its own cell, so Blatt 1 is bijective apart from the counted long-tail cases." CONTRADICTED AS STATED.** The v138 code assigns 722 source cluster types into 443 target cells, reusing cells. There are 147 collision cells containing 426 source cluster types; 1,047 of 91,234 non-empty-cluster encoding events used a collision-affected cluster (1.15%).
3. **"AUC below 0.85 means the generated text cannot be reliably distinguished from VMS-B." NOT VALIDATED.** A matched negative that scrambles the order of real Voynich tokens within each line passes this threshold in all 16 runs. This classifier measures one h2-based feature, not an encompassing sequence distribution or mechanism.
4. **"Plaintext is reversibly encoded, apart from counted prefix losses." NOT DEMONSTRATED.** Input spelling is non-injectively normalised, most source word boundaries are not explicitly represented, and token keys/repair steps introduce further ambiguity. No independent inverse decoder or end-to-end round-trip evaluation was supplied.

## Verified outcomes

| Check | Result | Status / limitation |
|---|---:|---|
| Supplied generator rerun and comparison to reference | Identical SHA-256: `43b3aa4b9a6709bdc86e6ad37c5be32477d462a68809f8c5cf2e1dd19c633cc2` | PASS; repeated run also identical |
| Supplied VMS-B h2 (120-token blocks) | 1.968690 bits | Reproduced |
| v138 generated h2 | 2.062432 bits | Reproduced |
| h2 gap | +0.093742 bits | Descriptive; not mechanism identification |
| Original h2 classifier AUC | 0.703968 | Reproduced; above chance |
| Output vocabulary in first 21,522 tokens | 3,084 types; 51.4% hapax | Reproduced; VMS-B reference materially more diverse |
| Final tokens with unattested atomic bigrams | 2,365 / 101,297 = 2.335% | FAIL of stated invariant |
| Final tokens with unattested atomic trigrams | 7,259 / 101,297 = 7.166% | FAIL of stated invariant |
| Nonempty-cluster events whose expected carried key != actually written cell key | 17,552 / 91,234 = 19.24% | Key propagation inconsistency; not a direct percentage of wrong decoded words |
| Initial seam candidates failing admissibility | 11,608 / 91,234 = 12.72% | Measured before fallback |
| Seam candidates still failing after repair attempts | 6,545 / 91,234 = 7.17% | Repair is incomplete |
| Final tokens failing the combined ngram `ok()` predicate | 7,269 / 101,297 = 7.176% | Includes post-processing and line markers |
| Prefix-class fallback counter | 1,791 / 13,653 long junction events = 13.12% | Matches reported loss counter, but **not** a complete loss budget |
| Cross-source core rank table stability (v138 vs v140 Hartlieb) | 297/300 shared cluster types assigned different cells | Core rank table rebuilt from each full source; this is **not** a same-code, frozen-table transfer test |

The final bigram/trigram violation counts are **not additive**: a token can violate both. Distinct measurement predicates explain why the 7,269 `ok()` failures need not equal the 7,259 trigram violations.

## Negative and positive controls for the h2 AUC gate

**Negative mechanism control:** For each of 16 seeded trials, independently shuffle the interior tokens of *each real VMS line*. Preserve the first and last token, line membership, token vocabulary and token frequencies. This leaves section and folio membership intact, but destroys much of the sequence ordering. The resulting AUC-vs-original-VMS distribution is **0.607507 mean, SD 0.002294** (range 0.602837–0.611535). This negative passes the proposed AUC < 0.85 criterion in **16/16** trials. Generated v138 AUC is **0.703968**, a difference of **+0.096461 AUC** from the negative mean, whose between-replicate SD is **0.002294**. The difference is descriptive; the scramble null is a sensitivity control, not a formal sampling model for distinguishing production mechanisms.

**Coarse positive control:** Shuffle atomic units within every reference token while retaining its original atom multiset and token positions. This visibly disrupts local glyph statistics and yields **AUC 1.000** against the reference. Accordingly, the instrument rejects a coarse defect but accepts a substantial order-destruction control.

**Interpretation:** The AUC threshold has no demonstrated specificity for the manuscript's ordered production process. An AUC of 0.704 is also not evidence of statistical indistinguishability; it represents measurable discrimination by this one feature.

## Source and reversibility issues

- `Paket_v100/vbm_tafel_v119det.py` uses lossy `norm_word`: `liebe` and `libe` both become `libe`; `meine` and `mene` both become `mene`; `vnd` and `und` are the same special `§` input. Thus recovery of the original orthography is impossible without side information.
- Source words are concatenated into an unspaced stream. The long-cluster boundary rule explicitly encodes 13,653 junction events, against 68,395 boundaries between normalised source words. This does **not** prove the remaining 54,742 boundaries are all undecodable, but it precludes treating the source word-boundary ledger as fully preserved by a stated one-to-one rule.
- `RANK` is populated by descending frequency of the **whole plaintext used for that run** and matched against descending frequency of all supplied VMS-B cells (`vbm_tafel_v138.py`, lines 30–33). Consequently the table is fitted to both the input and evaluation corpus; it is not a source-independent historical key or a held-out generalisation result.
- The expected next key `key` is carried through transitions, but most nonempty cluster events write `RANK[Kc][0]` without ensuring equality (`vbm_tafel_v138.py`, lines 115–140). The measured 19.24% disagreement weakens a claimed unique vowel-bridge inverse.
- The code attempts up to 40 alternative prefixes before adjusting the vowel carrier and potentially dropping prefix material (`vbm_tafel_v138.py`, lines 142–152). The counted 1,791 fallback events therefore cannot be interpreted as the **total** information loss.
- `belegung.json` assigns bridge pairs to vowels, but mapping one visible pair to one vowel is not equivalent to demonstrating that the *final* emitted stream preserves each source vowel under key mismatch and seam suppression.
- No executable inverse decoder, full normalised-input recovery accuracy, exact-match word rate, or calibration of the author's proposed 60%-of-table threshold was supplied. The 60% requirement is therefore **unverified**, not an established recoverability frontier.

## Inventory, section fairness and leakage

The supplied reference contains 39 distinct folio numbers spanning Herbal B (26–66), the biological/balneological range (75–84), and the recipes range (103–116). The implementation pools all reference lines to build target cells, adjacency counts and prefix sets, and tests the resulting output on the same pooled material. Odd/even type-coverage splits are not a holdout for the rank tables because the complete reference is used earlier. The generator has no labelled source segments corresponding to these distinct Voynich sections, and pooled performance does **not** establish within-section performance. Pharmaceutical label text must not be treated as interchangeable with running Stars or Recipes text.

## Optional v140 check

`HOMK=0.1 python vbm_tafel_v140.py` executes successfully with the supplied Hartlieb file (18,905 pre-panel tokens, 345 consonant-cluster types). The `panel_v126.py` comparison shows an especially concentrated conditional pair distribution: 0.26 bits versus 1.31 bits in VMS-B, with top-pair mass 95% versus 67%. This is not a section-matched test. The general `mess_v118.py` comparison **raises `ValueError`** for this shorter output because it tries to draw 180 synthetic blocks without replacement while fewer are available. This is an instrument edge-case defect, not a conclusion about v140.

## Qualification / discriminating test required before claims of VBM fidelity

1. **Freeze the complete encoder and all lookup tables.** Train cell, prefix, join and vowel tables on one section-matched discovery partition. Do not rebuild from test plaintext or held-out Voynich.
2. **Demonstrate actual inverse decoding** on independent German passages, with side channels prohibited. Report lossless recovery of the explicitly declared intermediate representation, ambiguity set sizes, and recoverability by section, rare-cluster bin, junction type and length. Declare all normalisation losses separately.
3. **Positive controls:** encoded texts with planted keys and known ground truth should be accurately decoded and recognised by the proposed instrument under fresh plaintext and permitted historical orthography variants.
4. **Hard matched negatives:** section-conditioned slot generators, first/second-order Markov, within-line shuffled order, and alternate source-language plaintexts. Match frequencies, token lengths, line structure, section and manuscript transcription. Require a precommitted discrimination criterion that rejects the hard negatives while accepting the planted positives.
5. **Blind, within-section transfer:** compare untouched Voynich validation material with fresh encodings, using predictive log-loss of multiple held-out sequential statistics (lag-1/lag-2, long-range repetition, word types/hapaxes, boundaries and opener effects), not one selected h2 AUC. Explicitly quantify uncertainty and avoid corpus pooling.
6. **Mechanism identifiability:** even a perfect blind surface match identifies a candidate **renderer**, not an underlying language, recovered plaintext, or historical encoder. An independent source-to-output anchor is required to identify the underlying mechanism.

**Verdict.** v138 is a computationally reproducible, deliberately fitted German-to-Voynich-like generator. As delivered, it fails two of its express implementation invariants, does not demonstrate end-to-end recoverability, and is not qualified by a discriminating held-out test. This closes the **current v138 evidential claim**, not the entire VBM hypothesis. A stronger architecture remains logically possible but is unconfirmed here.

## Reproduction

Extract the original supplied ZIP. From its `gen138` directory, run:

```sh
python vbm_tafel_v138.py
cmp GENERAT_v138_kurzertext.txt GENERAT_v138_kurzertext_REFERENZ.txt
python Paket_v100/mess_v118.py vms_B_REVA_atomstream.txt GENERAT_v138_kurzertext.txt
python panel_v126.py vms_B_REVA_atomstream.txt GENERAT_v138_kurzertext.txt
```

The independent audit scripts in the companion audit bundle are `audit_stage1.py` (mapping collisions/ngrams), `audit_stage2.py` (instrumented seam/key trace), `audit_stage3.py` (entropy scrambling), `audit_stage4.py` (negative/positive classifier controls), and `audit_stage5.py` (source-to-source rank mapping). Each saves its derived result atomically to a distinct `.pkl` checkpoint. The bundle also includes `audit_summary.json` with all verified numerical outcomes, and does not modify the original supplied ZIP.
