# VBM falsifiability and identifiability contract
**14 October 2026 | Status: PROPOSED CONTRACT; HISTORICAL TEST NOT ADMITTED**

## RETRACTED / NARROWED FINDINGS — keep at the top

1. **v142b did *not* implement the defining cross-EVA-space VBM architecture.** Its exact recovery from 68,396 development and 13,232 Hartlieb words is valid as an engineering demonstration of a *related in-word bridge cipher*, but its bridge pairs are placed **inside** cipher words, and its printed spaces reproduce source word boundaries. The author's VBM instead puts vowel bridges **across EVA spaces**, which need not represent plaintext word boundaries. The previous description "restores the author's bridge-pair logic" is true only of the lookup *values*, not their original positioning or hypothesis. **Do not cite v142b as a positive control for historical VBM.** [E1, A1]
2. **v138's reversible / attested-sequence claims remain rejected as implemented.** The original inventory has 722 complete consonant cluster types and 443 cells, including 147 collision cells; the supplied audit records 7,259 of 101,297 generated tokens with an unattested atomic trigram. The underlying code was reproducible but the specified invariants failed. Those figures are deterministic; no null SD is applicable. [A1, A2]
3. **A self-reference trigram-support diagnostic is not a mechanism discriminator.** Reference Currier B has 0 unattested trigrams against its *own* full inventory by definition. The v142b comparison establishes failure of a frozen engineering support rule, not a calibrated probability against VBM; shuffling intact tokens has null SD=0 under that support statistic. [A1]
4. **The v142b reversal does not restore the earlier V10 key-recovery failures, and the Q65 cross-boundary dependencies do not assign vowel values.** No evidence has been promoted from a structural pattern to medieval German plaintext. [P1]

## 1. A precise statement of the research question

**Historical hypothesis H-P:** A source text was transformed into Voynich glyphs using a reusable, table-based, position-dependent encoder. In the core VBM description, the left and right sides of an EVA space together encode a vowel, while token-internal slots encode consonant material; original word/syllable divisions need not coincide with EVA spaces.

**Restricted null H-S:** A manuscript-section-conditioned autonomous slot/Markov generator produces Voynich surface structure without encoding that source text. This is a *family*, to be constrained fairly only after H-P is fully specified.

**Unrestricted null H-S\*:** For any plaintext distribution and encoder, an autonomous process can sample directly from the same resulting ciphertext distribution. H-S\* is **exactly observationally equivalent** to H-P when only ciphertext is observed; no ciphertext-only statistic can distinguish them. This is an identifiability theorem, not an empirical result in favour of H-S\*.

**Testable engineering hypothesis H-E:** A specified encoder/decoder preserves its declared normalised source. Known source/cipher pairs can distinguish H-E from faulty encoders. H-E succeeding does not identify H-P historically.

**Question to settle:** Does a *fixed, restricted* H-P make a prediction tied to an independently known source or held-out input that H-S cannot make with the same information and complexity allowance? The answer is **not yet established**.

## 2. Claimed features, separated by evidential status

| Feature | Status | Why it matters / what would be lost by relaxing it |
|---|---|---|
| EVA spaces divide two parts of a vowel bridge `VL.VR` | **Core author definition** | Essential; abandoning it turns VBM into a different in-token cipher [F1]. |
| Vowel/consonant roles across the token chain, with C1/C2 inside | **Core interpretive hypothesis** | Requires a fixed, observable parser and a mapping specification [F1]. |
| EVA spaces are not assumed plaintext word boundaries | **Core author definition** | v142b contradicts this; its exact source-word recovery cannot be transferred to original VBM [F1, A1]. |
| C1 simpler substitution; bridge homophonic; C2 more complex/polyphonic | **Public working mechanism** | Define role and allowed variability before tests [F1]. |
| German/Bavarian source language | **Preferred, not essential** | The author explicitly says his theory does not *require* Bavarian; do not turn German fit into a necessary prediction post hoc [F1]. |
| REVA orientation | **Current preferred implementation, not irreversibly fixed** | Author explicitly worked with both orientations; freeze before comparative tests [F2]. |
| 40 prefix forms, exact pair assignments, one cluster per cell, ranked frequency tables | **Implementation choices** | v138 may fail without falsifying all table ciphers; requires an explicit complexity and exception budget [A2]. |
| Line-start and line-end symbols encode resets or states | **Unconfirmed mechanism** | May make a falsifiable observation only when permissible states/transition rules are specified *before* seeing test folios [F1]. |
| A/B or Stars drift reflects cipher development | **Speculation, explicitly not a timeline** | Section differences also occur under non-cipher grammars; historical order must be independently established [F3]. |
| 60% table correctness needed for partial reading | **Unsupported numerical heuristic** | No verified independent decoder/recovery curve for that threshold. No acceptance test should use it [A2]. |

**Author-source qualifications:** June 7 post calls the theory a possible explanation, not a demonstrated decipherment; September 9–14 posts treat REVA, key values and numeric tables as developmental. The 14 October email explicitly says the generator is imperfect and VBM remains possible but unconfirmed. None constitutes acceptance of the frozen contract below. [F1–F3, U1]

## 3. Source-verified experimental inheritance (no new generator fit)

The supplied original ZIP was hashed and its cross-space bridge specification read directly. The v138 independent audit JSON, v142b precommitted admission contract, v142b results JSON, and v142b full report were cross-checked by `verify_admission.py`; `RESULTS.json` and four atomic pickle checkpoints preserve them. Original ZIP SHA-256: `e67fe89555f8d8b2e74218077bb186f4f6fec4e5444f0ee9c289750d1efd4033`.

- **v138:** mechanically reproducible, but original strict invertibility and seam claims fail. The old h2 AUC gate also accepted within-line shuffled Voynich as a negative control. Thus no mechanism inference. [A2]
- **v141/v142:** showed some *alternative* codecs can decode normalised German (including Hartlieb) exactly from frozen tables. v141 paid for it with artificial control tokens; v142b moved bridges within source-aligned cipher words and still failed the surface gate. **Neither is a qualified positive control for cross-space historical VBM.** [A1]
- **Q65/Q65b:** earlier work found real cross-boundary structure, but did not establish five plaintext vowels or a VBM-only prediction; the restricted additional terms did not earn mechanism promotion. [P1]
- **V10:** synthetic cipher-key recovery failed precommitted gates, even under favourable known-source simulations. No new plaintext recovery or language-search programme is admitted on that basis. [P1]
- **Super-Grammar and slot controls:** structure is real and modelled independently; that does not prove it is an autonomous generator. The existence of an alternative compatible structure prevents making a unique plaintext claim from structure alone. [P1]

## 4. Identifiability audit, in the required order

1. **Circularity:** Existing v138 cells and prefixes were selected from the same Currier B transcription used to measure resemblance. There is no independent within-section surface validation of *historical* VBM.
2. **Leakage:** The author's real-German training, preselected botanical crib suggestions, and data-dependent segmentation cannot be used as blind targets. The supplied Hartlieb text was held out of later table formation, but it was also supplied by the same author and does not authenticate Voynich's language.
3. **Confounds:** Source genre, scribal hand, Currier language, topic, folio format, line geometry, scribal conventions, transcriber/atomiser, source orthography, word frequency and token length can all create associations. Controls must match or stratify these, not absorb them into a global score.
4. **Matched nulls:** Within-line shuffled intact words cannot be the only negative; many token-internal scores remain identical. A hard alternative must preserve sectional composition, line positions, word-length distribution, marginal/low-order dependencies and the same feature-learning privileges, while lacking *a plaintext-to-cipher relation*.
5. **Control fairness:** A VBM model cannot learn both a key and a tokenizer on a test section while its alternative gets a frozen grammar. Match tuning budget, data access, source choices and model-description cost.
6. **Measurement degeneracy:** Zero-variance self-support nulls, close-to-saturation scores and entropy agreement do not identify a cipher. Use proper, nondegenerate predictions with a separately qualified statistic.
7. **Representation dependence:** EVA/REVA, ZL/ZLZI/other transcription, glyph atoms and line direction must be frozen for a result, with sensitivity reported *after* qualification. No mixed sections or labels/running text.
8. **Decision fragility:** A result that depends on a manually chosen crib, 40 mutable prefixes, a revised exception, or a changed cutoff is exploratory only. The null SD and effect size must accompany every headline comparison; if effect/SD <2, lead with **the metric does not resolve this**.
9. **Audit completeness:** Immutable source hashes, data masks, instrument controls, preregistered failure thresholds, all corrections and a top-of-file retraction list required. This audit fulfills the *identifiability decision*, not the absent future historical test.

## 5. Formal reason not to run another ciphertext-only discriminator yet

Let `M` be an unknown plaintext, `K` a cipher key, and `C` the observed ciphertext. A plaintext model induces

`P_P(C=c) = sum_{m,k} P(M=m,K=k) P(C=c | m,k)`.

An autonomous model `S*` can be defined with `P_S*(C=c) := P_P(C=c)`. Then for **every** computed statistic `T(C)`, its probability law under P and S* is identical. Hence no score based only on this ciphertext can decide the general historical claim. **No null standard deviation is relevant to this exact logical equivalence.**

This does *not* make all restricted hypotheses untestable: an externally fixed VBM parser/key plus an independent anchor, or a tightly constrained counterpart generator incapable of learning that relation, can yield distinct predictions. It does mean that a flexible fitted slot null cannot be defeated merely by matching more features, and a flexible VBM cannot earn a historical pass by doing so.

### Candidate observables and admission decisions

| Proposed observable | Can discriminate? | Disposition |
|---|---|---|
| More accurate h2/Zipf/hapax/repetition/word lengths | No unique implication of plaintext | **STOP**, already trialled / circularity risk |
| Stable cross-EVA bridge distribution / C1–C2 dependence | Detectable structure; mechanism not exclusive | **STOP** as historical discriminator without independently fixed mapping |
| Independent exact decode of freshly VBM-encoded German | Distinguishes defective vs working cipher only | **Allowed engineering**, but not worth another generator iteration now |
| Robust recovery of *unknown* historical codebook by language model alone | In principle restricted test, but earlier synthetic V10 gates failed | **STOP** until a changed solver earns new planted-key qualification |
| Independently identified, securely localized plaintext/cipher correspondence, then frozen key predicts another disjoint source passage | Potentially yes against controls denied that relation | **GO: seek and register evidence; experiment only once anchor independently verified** |
| Secure contemporary VBM table or bilingual cipher source | Potentially yes, subject to authentication and dating | **GO: archival anchor search, without presumed discovery** |

## 6. What counts as an *independent anchor*?

An anchor must be identified **without** consulting VBM candidate plaintext output or fitted cipher values. It needs documented folio, transcription, provenance, textual relationship, and a way of ruling out equally plausible corresponding passages. A botanical picture alone, a medieval text from the same genre, or the word “hazel” *inferred* from a plant after a proposed reading does **not** qualify. The author’s f8v `nuzze zezzen` parallel is an interesting post-hoc candidate, not an independent crib. Zodiac month labels should not be pooled with running text; pharmaceutical labels are separately handled. De Balneis/Taccola illustration correspondences do not establish accompanying text correspondence.

Anchor register template: independent discovery date; manuscript witness and catalogue/IIIF; exact passage and transcription certainty; folio and textual genre; how the alignment was discovered blind to VBM; competing passage counts; whether a single small fixed key predicts another held-out contiguous passage; author/key blinding and independence risks. A null result remains visible.

**As of this audit:** no candidate in the reviewed material qualifies. This says nothing about whether one exists outside that material; further source research would be required.

## 7. Proposed author-facing freeze (requires the author's explicit responses)

**Do not treat this proposal as the author's accepted hypothesis.** Ask JoJo to declare:

1. **Core:** Is each ordinary *running-text* EVA gap necessarily a vowel bridge, or only a quantifiable subset? What exact exceptions and line resets are allowed? What proportion and how determined without plaintext?
2. **Codebook:** Which mappings of `(VL, VR)` to vowel and of `(C1,C2,context)` to consonantal unit are stable within a section? Which can vary, and by what *observable*, fixed rule? Provide frozen parser and decoding procedure.
3. **Recoverability:** Is the intended output a reversible normalised plaintext stream, a lossy phonetic representation, or merely Voynich-like ciphertext? If lossy, define information loss and the equivalence classes *before* recovery tests.
4. **Direction/section scope:** Which orientation(s) are proposed, and are Currier A, Currier B and each substantive running-text section expected to share one key or different predeclared keys? Preserve separate holdouts, no section pooling.
5. **Falsifier:** Specify one result on a fresh synthetic positive and one result on an independent historical anchor that would cause him to withdraw a *restricted* VBM variant. Set the exception/complexity ceiling before testing.

Absent answers 1–5, no historical-data model fit is authorised. The contract can be revised *prospectively*, but failed old variants remain failed, with same-prominence retractions.

## 8. If an independent anchor emerges: one prospective test, no fishing

*Only a protocol skeleton is admissible until the anchor and fixed model exist.*

- Pre-register a **single running-text section and specific passage**, distinct from all fitted examples; define folio/quire holdout groups and source variant uncertainties. No mixing Currier A/B, Herbal/Stars/Pharma or labels with running text.
- Register the anchor independently with its competing correspondences before accessing cipher output. Freeze direction, transcription, parser, tables, all exceptions and the source normaliser.
- Require planted positive text/cipher pairs of comparable length, genre and allowed orthographic noise. The instrument must recover the registered relation and reject deliberately false anchors and realistic alternative-language/source controls.
- Hard negatives: section/hand/length/line-matched autonomous slot generators with equal fitting privilege; false passage alignments with identical search budget; frequency/Markov controls and label shuffles only where causally relevant. Preserve word order for any test intended to measure word order.
- Primary statistic: *held-out log predictive score of the anchored continuation given the preregistered bridge/key*, compared against all competitors (with full coding/selection costs). Define its measurement and exact expected effect **before observing the withheld passage**. Include null mean, null SD, effect size, permutation max-statistic and multiplicity control. Preliminary promotion only if the effect is at least 2 null SD **and** a familywise calibrated test rejects at 0.05. These are necessary, not sufficient, for attribution.
- Repeat only as a separately registered replication on a second independently anchored passage *within the same section* or other explicitly predeclared domain. A source match without an independent anchor remains a descriptive correspondence.

If the positive control fails: **INSTRUMENT FAIL**, no Voynich inference. If positives pass but hard negatives cannot be separated: **UNIDENTIFIABLE**, no Voynich inference. If source linkage survives: evidence supports **the restricted encoding hypothesis**, not automatic German/Bavarian attribution or complete decipherment.

## 9. Four adversarial verification questions

**Q1. Could v142b's exact decoding establish that historical VBM is feasible?** No. It is reversible under its own grammar, but the bridge location and spaces contradict the author's defining claim. It is a different formal object.

**Q2. Could a cross-gap feature outperform one small slot null without independent plaintext?** Yes, a *restricted* comparison might favor it if frozen and blinded. But it cannot identify plaintext encoding against an equally expressive autonomous alternative; our prior Q65 results already caution against promoting first-order bridge dependencies.

**Q3. Is the f8v botanical parallel an anchor?** No. The description and reading are mutually selected. A corresponding passage identified and authenticated *independently* could change this, but that evidence is absent here.

**Q4. Does the current STOP rule declare VBM false?** No. It declares the available historical *identification problem* unresolved and further unanchored fitting unproductive. v138 implementation-specific falsifications stay in force; the broad mechanism is neither proven nor ruled out.

## 10. Final admission decisions

| Workstream | Decision | Condition to reopen |
|---|---|---|
| Tune v138/v142 generator to more surface statistics | **STOP** | New, independently registered and discriminating observable |
| Historical Bavarian/plaintext claims | **NOT IDENTIFIED** | Anchor and frozen mechanism with qualified controls |
| Author's formal mechanism clarification | **GO** | Obtain five explicit commitments above |
| Independent archival/textual anchor triage | **GO (research only)** | Independently verified passage and negative candidate set |
| New ciphertext-only Voynich hypothesis test | **NOT ADMITTED** | Fixed model + exclusive observable or external anchor |
| Preserve v138–v142 failures and code | **REQUIRED** | No overrides; corrections at report top |

**Bottom line:** The next result worth producing is not another ciphertext sample. It is an independently evidenced correspondence that forces VBM to predict unseen text, or an author-frozen constraint that an equally empowered nonplaintext process cannot reproduce. Until then, further fitted similarity has no mechanism-identifying value.

## Evidence inventory and source links

- **[U1]** User-provided JoJo message dated 14 Oct 2026 (current conversation), reporting VBM unconfirmed; numeric 60% decoding heuristic without reproducible derivation.
- **[F1]** JoJo, 7 June 2026, definition and mechanism: https://www.voynich.ninja/archive/index.php/thread-5312-25.html
- **[F2]** JoJo, 9–15 Sep 2026, reading direction, internal tables: https://www.voynich.ninja/archive/index.php/thread-5312-41.html and https://www.voynich.ninja/archive/index.php/thread-5312-43.html
- **[F3]** JoJo, 2 Sep 2026, on Stars/section drift as speculation: https://www.voynich.ninja/archive/index.php/thread-5312-40.html
- **[A1]** `VBM_v142_Admission_Report_2026-10-14.md`, `03_pair_bridge_contract.json`, `05_pair_bridge_results.json`; supplied v142 reproducibility bundle.
- **[A2]** `Generator_v138_Pruefpaket_2026-10-14.zip` (`gen138/README.md`, `vbm_tafel_v138.py`) and `VBM_v138_Independent_Audit_2026-10-14.md`, `audit_summary.json`; supplied original and audit.
- **[P1]** Previous conversation findings Q65/Q65b, V10, Super-Grammar, as preserved in research history. Raw original Q65/V10 run files were **not re-opened in this turn**; these are prior findings, not newly reverified numeric results.