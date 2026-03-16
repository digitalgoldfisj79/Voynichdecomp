# A Two-Table Cipher Architecture for Beinecke MS 408: Function Word Assignments from Cross-Validated Pharmaceutical Latin

**Edward Bozzard**

Independent Researcher · ORCID 0009-0002-4052-0994

edwardbozzard@gmail.com

**Target:** *Cryptologia*

**Data availability:** DOI 10.5281/zenodo.18812705

---

## Abstract

The Voynich Manuscript (Beinecke MS 408) has resisted decipherment for over a century. In a companion paper (Bozzard 2026a), we characterised the manuscript's text as following a four-slot positional grammar (PGCS) with a copy-mutate production signature, but left the production method open.

This paper proposes a specific cipher architecture, a two-table system combining a function-word nomenclator with a content-word syllabic grid, and tests it against external pharmaceutical Latin corpora. Using a greedy frequency-matching algorithm with plaintext candidates drawn from an independent fourteenth-century pharmaceutical Latin manuscript (Ms.Ald.211) and optimised against the manuscript's suffix-family bigram distribution, we infer candidate suffix-family assignments for ten Latin function words. The assignments reproduce the manuscript's bigram structure at r = 0.96 on the training corpus, cross-validate at r = 0.89 on Circa Instans (unseen during training), hold on three held-out manuscript folds (r = 0.91–0.95), and exceed all 10,000 random assignments (p < 0.0001). A leave-one-out analysis confirms that the conjunction *et*, routed to the Y-family, accounts for 76% of the training improvement; this assignment was independently established from manuscript-internal evidence. Nine of ten assignments disagree with the vowel heuristic, consistent with a dual architecture: a systematic cipher grid for content words and an arbitrary code table for function words. This architecture is assembled from components individually attested in fifteenth-century Northern Italian cryptographic practice; the full integrated form is directly documented in Amadi's treatise (c. 1570s).

As a directed corroboration test, the assignments produce a selective consonant-vowel (CV) enrichment on folio 2r, previously identified as Centaurea from illustration analysis: the syllable 'mi' (hypergeometric p < 10⁻⁶, surviving Bonferroni correction across ~960 folio-CV tests), consistent with *minoris*, the qualifier distinguishing Centaurea minor in pharmaceutical Latin. Five additional Herbal-A folios show Bonferroni-surviving CV enrichments consistent with pharmaceutical vocabulary. A forward cipher implementing this architecture, with scribe parameters calibrated against the manuscript, scores 62.9/84 on a distributional battery and was evaluated on all nine manuscript sections without modification. We report one false positive and a systematic record of killed hypotheses (§9).

**Keywords:** Voynich Manuscript, nomenclator, babuini cipher, syllabic substitution, cryptanalysis, PGCS grammar, cross-validation

---

## 1. Introduction

### 1.1 What Paper 1 established, and what it left open

Bozzard (2026a) showed that 92.96% of the Voynich Manuscript's 37,465 tokens conform to a four-slot positional grammar: Prefix–Gallows–Core–Suffix (PGCS). The text exhibits a copy-mutate production signature: 83.9% of tokens match a nearby predecessor on at least two of four PGCS slots, with novel tokens appearing 2.0× more often at line beginnings (hapax rate 24.4% line-initial vs 12.2% elsewhere, full manuscript), consistent with a scribe consulting a reference at line breaks.

The best generative model in that paper (Gen-SP) scored 59/84 on a distributional battery. The circular transcription model scored 67–76/84 (mean 71.3). The 12-point gap represented something the generator could not capture. We called it "content" and noted that it was "consistent with scribal composition, faithful copying, or a complex cipher."

This paper addresses the cipher hypothesis. If the manuscript enciphers a natural language, three questions follow: what kind of cipher, what language, and can we recover any of it? We provide evidence for the first two (a nomenclator-grid cipher on pharmaceutical Latin) and infer candidate function-word assignments for the nomenclator, with a directed corroboration test at the folio level.

### 1.2 How this differs from previous decipherment attempts

Every previous attempt starts from a language hypothesis and works inward: assume a plaintext, propose a mapping, evaluate the output. This approach has produced myriad contradictory claims spanning Latin (Strong 1945), proto-Romance (Cheshire 2019), Nahuatl (Tucker & Talbert 2014), and others. None has achieved scholarly consensus, because with enough degrees of freedom any mapping can produce apparently meaningful text from any source.

This paper works in the opposite direction. We begin with the manuscript's structural properties (PGCS grammar, EC/FC bimodality, suffix families) and ask what external data reproduces those properties. The assignment inference does not start with "assume Latin." It starts with "the VMS has these suffix-family bigram frequencies" and asks "which function word assignments reproduce those frequencies?" The language enters through corpus selection (pharmaceutical Latin was chosen for independent reasons documented in §8), but the specific assignments are the output of the optimisation, not predetermined.

Three specific methodological differences distinguish this work. First, separation between plaintext source and optimisation target: assignments are derived from external corpora (Ms.Ald.211, Circa Instans), while the optimisation target is the manuscript's own suffix-family bigram distribution. The manuscript defines what needs to be matched; the external corpora supply the candidates. This is a manuscript-constrained model-selection exercise, not an unconditioned external recovery. Second, falsifiable enrichment tests: the CV reader produces specific, testable claims about folio content. Third, systematic reporting of negative results: several alternative hypotheses were tested and rejected (§9).

---

## 2. Cipher History in Northern Italy

### 2.1 Components attested before the manuscript

The Voynich Manuscript's parchment is radiocarbon dated to 1404–1438. Every component of the cipher architecture we describe is individually attested in Northern Italian practice within or near this window, though the full integrated system is first directly documented later (§2.2).

*Nomenclators* (dedicated cipher symbols for common words) were in use from the 1350s, when Gabriele de Lavinde of Parma compiled the earliest known nomenclator for Antipope Clement VII's secretariat (Kahn 1967). The Mantua 1450 cipher in the Tranchedino ledger includes a nomenclator with entries for *Come*, *Quando*, *Non* and other common words (Pelling 2016; Meister 1902).

*Syllable-group substitution* appears in the Mantua 1450 cipher, which includes arbitrary shapes for syllable groups (ab, ac, ad, af, etc.) (Pelling 2016). A Milanese cipher from 1448 also employed double glyphs for two-letter syllables (Meister 1902).

The Mantua 1450 cipher is the earliest known key combining nomenclator and syllable groups in a single system. However, its syllable groups are an unstructured inventory, not a grid. The grid organisation (consonant rows, vowel columns, with a keyword permuting row assignments) is first documented in Amadi's *Trattato delle cifre* (c. 1570s), though Amadi describes it as established practice.

### 2.2 What Amadi describes

Amadi's treatise (Scheers 2020 edition) catalogues over 90 cipher methods. The babuini sections (§§0074–0078) describe a consonant-vowel syllable grid with keyword variation, multiple cipher alphabets ("houses"), a nomenclator for common words, and worked examples showing step-by-step encryption and decryption.

Three features of Amadi's descriptions are critical for interpreting the VMS.

First, *scribe choice is integral to the cipher, not an afterthought.* The variation table (§0076, 037v) provides over twenty cipher options per consonant row. Amadi's instruction for encryption is explicit: "on that line choose any number from the line and write it down" (Scheers 2020). The cipher designer supplies the grid; the scribe chooses which cell entry to write for each syllable. This means the statistical surface of a babuini cipher (its word frequencies, repetition patterns, and vocabulary diversity) is determined by the scribe's selection behaviour, not by the grid contents alone. The grid constrains what can be written; the scribe determines what is written.

Second, *the keyword permutes only the consonant-row assignments.* The babuini table (§0074, 036v) shows a fixed grid of syllables (ba, be, bi, bo, bu / ca, ce, ci, co, cu / ...) with the keyword controlling which transposed letter maps to which consonant row. The vowel columns remain fixed. Changing the keyword rearranges which consonant family each cipher symbol represents, but does not alter the grid's internal structure. This is exactly the architecture we infer: a fixed vowel-to-suffix-family mapping with a permutable consonant-to-row mapping.

Third, *Amadi explicitly separates the grid from the nomenclator.* The worked examples (§§0075–0077) show common words enciphered through dedicated code entries (the nomenclator) while content words pass through the syllable grid. The nomenclator entries are arbitrary assignments, not derivable from the grid rule. This dual architecture (systematic grid for content, arbitrary code for function words) is what we infer in §5.

Amadi also proposes reducing the Italian alphabet to 12 characters (Scheers 2020), which shows that Venetian cryptographers actively thought about small cipher alphabets; the manuscript's functional character inventory is approximately 15.

### 2.3 Independent convergence with Greshko

We engage with Greshko (2025) at length because it is the most recent published cipher model for the VMS and the closest to our approach. Greshko reached compatible conclusions from a different direction. His Naibbe cipher, a verbose homophonic substitution system, replicates the manuscript's character entropy and Zipfian structure but lacks long-range correlations and CV-syllable structure. His §4.2 identifies scribal habits as the likely cause of these long-range patterns; our copy-mutate scribe provides this. His §4.3 proposes deterministic CV-syllable respacing; our babuini grid provides this. His §5 conjectures four properties of a hypothetical VMS cipher; our model is consistent with all four. The historical plausibility argument also converges: Greshko independently concluded that a verbose substitution cipher of this kind requires no concepts unavailable in fifteenth-century Northern Italy.

### 2.4 The σ constraint

The manuscript's word-length standard deviation σ = 1.72 is below every character-level cipher in the Bowern-Gaskell (2022) corpus (minimum σ = 2.37, No_Vowels cipher). Character-level operations preserve source word-length variance (Latin σ ≈ 2.5); syllabic operations compress it toward the mean syllable count. This arithmetic constraint, requiring no assumptions about plaintext language, eliminates the entire character-level cipher class and places the VMS in the syllabic or word-level category.

---

## 3. The PGCS Foundation

This section is deliberately brief; the full treatment is in Bozzard (2026a).

The PGCS grammar decomposes every manuscript token into four positional slots: Prefix, Gallows, Core, and Suffix. Tokens divide into two populations: **empty-core** (EC, 52.7% of text) with no core content, and **filled-core** (FC, 47.3%) with a non-empty core string. For example, the common token *daiin* decomposes as prefix 'd' + suffix 'aiin' with no core (EC), while *chor* has prefix 'ch' + core 'o' + suffix 'r' (FC). EC tokens are shorter, more frequent, and repeat heavily; FC tokens are longer, rarer, and generate most hapax legomena.

### 3.1 Transcription basis and robustness

The VMS text data derive from the ZLZI consensus transcription (Zandbergen-Landini, maintained at voynich.nu), which was decomposed into PGCS slots (prefix, gallows, core, suffix) and enriched with per-token metadata: section assignment, folio, line number, suffix family, m_core normalisation, and EC/FC classification. This enriched dataset (37,465 tokens, 9 sections, 224 folios) is the analytic base for all tests in this paper and is archived at Zenodo [DOI]. The PGCS decomposition is deterministic given the EVA token string; no manual annotation was applied. Suffix family assignment follows from the terminal character(s) of each token, as defined in Bozzard (2026a).

All analysis uses the European Voynich Alphabet (EVA) transcription standard. We tested robustness to transcription representation: collapsing EVA digraphs into single symbols (ch→C, sh→S) reduces character bigram coverage from 31.9% to 27.9%; the constraint becomes tighter, not looser, under glyph-level representation. Of the 20 most frequent Herbal-A tokens, none changes its PGCS decomposition or row assignment under known transcription variants (ZL vs Stolfi). The PGCS decomposition provides two core representations: raw 'core' (preserving digraphs) and 'm_core' (normalising digraphs: ch→c, sh→∅). Row-level frequency analysis uses m_core (7 rows); within-row consonant resolution uses raw core, where longer core strings encode less frequent consonants within a group.

---

## 4. The Two-Table Architecture

### 4.1 Function words and content words

When a Latin word enters the cipher, the first question is: is this a common function word or a content word?

**Function words go to Table 1: the nomenclator.** Each function word has a fixed code: an EC token whose suffix family identifies which word it encodes. Six suffix families exist (Y, N, L, R, BARE, M), named by their characteristic final characters. The token's prefix identifies which "house" variant the scribe selected, which provides polyalphabetic variation: the same function word can appear as four different-looking tokens across four houses. House 4 shows a distinctive collapse pattern where 7 of 16 suffix types reduce to the null suffix (p < 0.01 against permutation null), which suggests a simplified encoding in which some function-word distinctions are lost.

**Content words go to Table 2: the syllabic grid.** The grid has consonant rows and vowel columns. The word's initial consonant selects the row; its first vowel selects the column (and hence the suffix family). The cell at that intersection contains a core string. Together with a prefix, this produces a complete PGCS token.

### 4.2 Why two tables explain the bimodal vocabulary

No single-mechanism cipher produces two structurally different token types. The two tables do: function words bypass the grid (producing short, frequent EC tokens), while content words go through it (producing longer, more varied FC tokens). The six properties any cipher hypothesis must explain (four-slot structure, bimodal vocabulary, the σ = 1.72 constraint, section-specific profiles, character-level finite state machine, and vocabulary dynamics) are all addressed by this architecture.

### 4.3 The copy-mutate scribe

None of the historical cipher sources describe what happens after the grid lookup. The manuscript's text shows preferential reuse (83.9% of tokens match a recent predecessor), suffix avoidance (~55% on consecutive same-cell pairs), boundary-biased innovation (2.0× hapax rate at line beginnings vs elsewhere), and column stickiness (~22% probability of remaining in the preceding token's suffix-family column; the resulting same-family bigram rate is 0.252 versus 0.204 expected under independence). Column stickiness contributes to the manuscript's word-length autocorrelation (lag-1 r = +0.076, Z = 4.8 against shuffle null), because suffix families have different mean token lengths (BARE 3.4, N 5.7, Y 4.9 characters); the stickiness parameter was calibrated against the observed bigram rate (§6.2). The first three of these production habits are measured in Bozzard (2026a); column stickiness is new to this paper. These are the habits of a person writing by hand, consulting the grid at line breaks and copying from visual memory in between.

This copy-mutate production layer (Bozzard 2026a) is what separates the Voynich cipher from the clean table lookups that Amadi describes. The same Latin function word *et* produces different Y-family tokens (*chy*, *cthy*, *yky*, *shey*) through copy-mutate variation. Classical frequency analysis, which depends on stable cipher-to-token mappings, fails against this production model. The assignment inference in §5 succeeds because it operates at the suffix-family level, which is preserved through copy-mutate variation: the scribe varies the surface form but never changes the family.

---

## 5. Function Word Assignment Inference

The two-table architecture predicts that function words follow non-random suffix-family assignments set by the cipher designer, not derivable from the grid's vowel rule. If the plaintext is pharmaceutical Latin, those assignments should be inferable from independent pharmaceutical Latin corpora by matching function-word sequence statistics against the manuscript's observed patterns.

### 5.1 Corpus preparation

Both Latin corpora were transcribed from manuscript images by automated HTR, not from critical editions.

**Circa Instans (validation corpus).** Transcribed from Wellcome Collection MS 624 (ff. 1r-69r), a mid-fifteenth-century paper copy of the Circa Instans versio B, "redazione Göttingen" (Ventura 2015), containing 277 chapters. The manuscript is digitised under Public Domain Mark at wellcomecollection.org. Recognition used Transkribus Text Titan I ter (model ID 356425, v2.41.0, run 8 March 2026), a transformer-based supermodel reporting 9.6% CER on historical Latin-script documents including Latin (READ-COOP 2025); no custom fine-tuning was applied. Tokenisation: lowercase, punctuation stripped, numbers removed, whitespace-split, minimum two characters. Yield: 52,004 tokens, 14,206 types. The PAGE XML export is archived at Zenodo [DOI]. Spot-checking identifies 15 consonant-heavy tokens as probable HTR errors or unresolved abbreviation marks (e.g. *cccccus*, *cębrum*), and 663 line-end hyphenations (marked with ¬ in the HTR output) that were stripped but not rejoined during tokenisation, producing split tokens. Correcting the hyphenation changes the row distribution by less than 0.3% on any row (χ² 0.045 → 0.042) and does not affect any reported result. The distributional tests in this paper use initial consonant and first vowel of each word; mid-word character substitutions, which dominate HTR error, do not affect these features.

**Ms.Ald.211 (training corpus).** Transcribed from Gallica digital images of Pavia, Biblioteca Universitaria, MS Aldini 211, a fourteenth-century composite manuscript containing Pseudo-Apuleius herbarius material and Circa Instans / Platearius pharmaceutical entries. One entry cites a Florentine medical authority ("probatum per Magistrum de Flozetia"), placing it within the same Northern Italian pharmaceutical text network that produced the VMS's likely source material. The manuscript was selected for genre (pharmaceutical Latin), period (fourteenth century), and regional provenance (Northern Italy). It has no known textual relationship to the VMS. The transcription was produced by LLM-assisted reading (Claude, Anthropic) from the Gallica images and cross-checked against the BnF catalogue plant identifications. Yield: 2,006 usable Latin words after filtering, with approximately 670 EC tokens and 350 EC-EC bigram pairs. This is a small training corpus; the cross-validation on CI mitigates overfitting risk.

**EC/FC classification.** The 373-word EC set was constructed by frequency threshold to match the manuscript's observed 53/47 EC/FC ratio (§3). This threshold correctly classifies standard Latin function words (et, in, cum, de) but also includes high-frequency pharmaceutical content words (folia, herba, aqua, radix) which recur in nearly every CI entry. These words route through the EC heuristic path, not the grid, which is consistent with the cipher's design: any word above a certain frequency threshold benefits from the polyalphabetic variation that the EC path provides. Both corpora and the tokenisation pipeline are archived at Zenodo [DOI].

### 5.2 Optimisation target

The target is the EC-EC suffix-family bigram distribution of VMS Herbal-A: the frequency of each ordered pair of suffix families in consecutive EC tokens (with intervening FC tokens ignored). This distribution has 43 non-zero bigram types over 1,364 pairs and captures the sequential structure of function word usage: how often Y-family is followed by N-family, N by L, and so on. A Latin text whose function words are assigned to the correct families will reproduce this sequential structure; incorrect assignments will not.

### 5.3 Algorithm

Starting from two known assignments (et → Y and in → N, established in Bozzard 2026a from bare-Y distributional analysis and the *daiin* anchor-line method), the algorithm proceeds as follows. Each Latin word in the training corpus is classified as EC (function word) or FC (content word) using a frequency-based threshold matching the manuscript's 53/47 EC/FC ratio. (This ratio is a structural parameter derived from the VMS; the non-circularity claim applies to the function-word *assignments*, not to the EC/FC boundary itself. The boundary determines which Latin words are eligible for assignment; the assignments are then trained entirely on the external corpus.) EC words are assigned to suffix families: known words use their fixed assignment, and all others use the first-vowel heuristic (a→Y, e→R, i→N, o→L, u→BARE). The algorithm then iterates over all eligible EC words, restricted to genuine Latin function words (prepositions, conjunctions, pronouns, common verbs, adverbs), testing all six possible family assignments for each word, and selects the word-family pair that maximises Pearson r between the training corpus's predicted bigram distribution and the VMS target. That word is permanently assigned and the process repeats.

Of 23 eligible function words with frequency ≥ 3 in the training corpus, eight improve r by ≥ 0.002 and are assigned. The remaining 15 show either no improvement (est, per, si, item, non, aut, bene, qui, eius, sic) or negligible improvement below threshold (eam at Δr = 0.001). The algorithm terminates when no further word exceeds the threshold. The total search space is modest: 23 words × 6 families = 138 trials per iteration, with 8 iterations before convergence.

### 5.4 Result

Ten words, eight free parameters:

| Family | Assigned words | Latin word class |
|--------|---------------|-----------------|
| Y | et, postea | conjunction, temporal adverb |
| N | in, cum, hoc | prepositions, demonstrative |
| L | de, habet, uel, que, supra | prepositions, connectives |

The classification of *habet* as a nomenclator entry rather than a content word reflects its formulaic function in pharmaceutical Latin: in recipe texts, *habet* serves as a stereotyped connector ("X habet folia... habet flores... habet radicem") with reduced lexical content, analogous to "has" in English recipe templates. Its assignment to L-family reflects this formulaic usage, not a claim that *habet* is a function word in the grammatical sense.

The inferred nomenclator assignment is these ten words. The forward cipher engine (§6) additionally includes ad → L and vel → L (orthographic variant of uel), yielding twelve entries total. These two fall below the Δr ≥ 0.002 threshold and are engineering additions for the forward model, not part of the cross-validated inference.

Every word tested beyond these twelve, including est, non, quod, sunt, fiat, per, ut, qui, shows zero improvement when reassigned from its vowel-heuristic family. The vowel heuristic appears correct for R-family (e-initial words: est, eius, per, ex, sed) and BARE-family (u-initial words: quod, ut, qui): no word in either family improves by reassignment. This is not a limitation of the method; it means the cipher designer assigned these words to the families the grid rule would have assigned them to anyway. The nomenclator was needed only for the words whose first vowel did not match their intended family.

### 5.5 Verification chain

Five corroborative tests, each using different data or method:

**(a) Training fit.** Bigram correlation: r = 0.31 (heuristic only) → r = 0.96 (nomenclator). [Figure 1 shows the before/after bigram scatter.]

**(b) Cross-validation.** Applied to Circa Instans (unseen during training): r = 0.44 → 0.89. The improvement on an independent corpus of the same genre confirms that the assignments capture pharmaceutical Latin function-word patterns in general, not idiosyncrasies of the Ald.211 text.

**(c) Held-out VMS.** Three-fold cross-validation on Herbal-A folios (folios divided into three groups by position): r = 0.91, 0.95, 0.95. All three folds independently confirm the assignments.

**(d) Null model.** 10,000 random assignments of eight words to six families (with et → Y and in → N held fixed): 0/10,000 match or exceed our r. p < 0.0001. An expanded null drawing eight random words from the full 23-candidate pool and assigning them randomly also yields p < 0.0001 (1/10,000 exceeds our r, with null maximum 0.961). Assigning all 23 candidates randomly gives the same result (p < 0.0001, null maximum 0.966). As a language control, randomly assigning Italian function words (*e*, *di*, *con*, *per*, etc.) never exceeds r = 0.84, confirming the result is language-specific.

**(e) Leave-one-out.** Each of the eleven words (ten inferred plus eam) is removed and the training, CI, and held-out r values are recomputed:

| Remove | Δ train | Δ CI | Δ held-out |
|--------|---------|------|-----------|
| et | −0.513 | −0.392 | −0.547 |
| cum | −0.054 | −0.021 | −0.051 |
| de | −0.034 | −0.015 | −0.035 |
| habet | −0.016 | +0.005 | −0.014 |
| hoc | −0.008 | +0.003 | −0.006 |
| supra | −0.006 | −0.001 | −0.005 |
| postea | −0.005 | −0.002 | −0.005 |
| uel | −0.004 | −0.007 | −0.003 |
| que | −0.003 | −0.003 | −0.004 |
| eam | −0.001 | −0.000 | −0.001 |
| in | 0.000 | 0.000 | 0.000 |

The conjunction *et* accounts for 76% of the training improvement. The vowel heuristic routes *et* (first vowel 'e') to R-family, but the manuscript's Y-family is heavily enriched for the sequential patterns that *et* produces in Latin (noun-*et*-noun constructions). Correcting this single misrouting transforms the bigram profile. Removing *in* changes nothing: the vowel heuristic already routes *in* → N (first vowel 'i') correctly, so the nomenclator entry is redundant. [Figure 2 shows the leave-one-out Δ values.]

The remaining eight words are nonetheless essential. With only the two known assignments (et → Y, in → N), training r = 0.81, which falls below the null maximum of 0.953. The eight free assignments raise r from 0.81 to 0.96, crossing the significance threshold. On the independent CI corpus, the eight words improve cross-validation from r = 0.84 to r = 0.89. The nomenclator is not a one-word result: *et* provides the largest single correction, but statistical significance and cross-corpus generalization both depend on the full table.

A necessary caution: training r alone does not discriminate source languages. A greedy optimizer applied to German pharmaceutical text (BSB Cgm 384, 801 words) achieves r = 0.98 by assigning word fragments and nouns to families — higher than our Latin r = 0.96. With eight free parameters and fewer than 200 EC-EC pairs, the optimizer can overfit any language's function-word distribution to the VMS target. The operative discriminant is cross-validation on an independent corpus of the same language: Latin validates at r = 0.89 on 52,004 unseen CI words; German has no comparable second corpus, and the assignments it produces (*fu* → L, *haffen* → R) are not generalisable function words. The training r confirms internal consistency; the cross-validation confirms the language. We predict that German cross-validation on an independent corpus would fail, because the assigned words are corpus-specific fragments rather than generalisable function words; this prediction is independently testable but is also moot, since §8's sonorant concentration test (93.3% vs German 44-50%) eliminates Germanic languages without reference to the bigram analysis.

### 5.6 Cipher versus code

Nine of ten inferred assignments disagree with the vowel heuristic. Only *in* agrees (first vowel i → N, nomenclator also N). This pattern is consistent with a dual architecture:

The **FC grid is a cipher**: systematic, rule-based. The first vowel of the Latin content word determines the suffix family. The vowel heuristic IS the grid rule. A scribe who knows only this rule can encipher any content word correctly.

The **EC nomenclator is a code**: an arbitrary lookup table. Each function word is independently assigned to a family by the cipher designer, not derivable from any spelling rule. A scribe who knows only the vowel rule would misroute every function word except *in*.

This dual architecture (systematic cipher for content words, arbitrary code for function words) is consistent with the structure described in Amadi's babuini cipher treatise and in the Mantua 1450 cipher key. The distinction between cipher and code within a single system is the theoretical contribution of this paper.

### 5.7 L ≡ N explained

L-family and N-family tokens have statistically indistinguishable successor profiles (χ² = 5.6, p = 0.47, seven successor categories, EC-EC same-line pairs across full manuscript), implying both families encode the same syntactic class. The assignment inference explains this independently: N = {in, cum, hoc} and L = {de, habet, uel, que, supra}: both families contain prepositions followed by nouns in pharmaceutical Latin formulas (*in aqua*, *cum melle*, *de cortice*, *ad dolorem*). Three methods now converge on the same conclusion: successor profile analysis, assignment inference (this paper), and the instability of cum/de between N and L across different optimizer threshold settings.

### 5.8 A worked example

With the nomenclator assignments now established, we trace seven Latin words through the two-table system to show how the architecture produces VMS-like text. The actual grid contents are unknown; this uses the identity permutation (no keyword scrambling) for exposition. The EC/FC classification is frequency-based, not semantic: high-frequency content words (folia, herba, aqua) are classified as EC because they exceed the frequency threshold, while lower-frequency content words route through the grid.

**Step 1 - deterministic lookup.** Function words and high-frequency words go to EC pools by family; lower-frequency content words go to the grid (§4.1). The grid assigns a row by initial consonant and a column (suffix family) by first vowel.

| Latin word | Route | Lookup | Raw token |
|---|---|---|---|
| *centaurea* | Content → grid | c → row 'o'; vowel 'e' → R-family | *chor* |
| *habet* | Function → nomenclator | → L-family; prefix 'ch' | *chol* |
| *radicem* | Content → grid | r → row 'd'; vowel 'a' → Y-family | *dy* |
| *et* | Function → nomenclator | → Y-family; prefix 'ch' | *chy* |
| *pulverem* | Content → grid | p → row 'o'; vowel 'u' → BARE | *cho* |
| *in* | Function → nomenclator | → N-family; prefix 'd' | *daiin* |
| *cortice* | Content → grid | c → row 'o'; vowel 'o' → L-family | *chol* ← duplicate of *habet* |

Raw output: seven tokens, only six distinct. The grid maps *cortice* (c-initial, first vowel o) to L-family, the same family as the nomenclator's *habet*. The same surface form *chol* can encode either a function word or a content word; only the cipher key distinguishes them.

**Step 2 - scribe variation.** The scribe applies copy-mutate rules (§4.3): vary prefixes to avoid repetition, vary suffixes within families, innovate at line breaks.

> *chor · chol · dy · chy ·* **sho** *· daiin ·* **shol**

The duplicate is resolved by prefix variation (*ch-* → *sh-*). The seven final tokens are all distinct and all attested in the manuscript's Herbal-A section.

---

## 6. Forward Cipher Validation

The nomenclator's value can be measured by its effect on the forward cipher: a generative model that takes real Latin source text as input and produces synthetic VMS-like tokens through the two-table architecture.

### 6.1 What v11 is and what it is not

The forward cipher v11 (Supplement S1) classifies each Latin word as function or content, routes function words through the nomenclator to EC tokens and content words through the grid to FC tokens, applies copy-mutate scribe rules (preferential reuse, suffix avoidance, boundary innovation, derived from Bozzard 2026a), and validates all output against the PGCS character-state machine. The cell pools from which v11 draws tokens are VMS Herbal-A vocabulary grouped by observed (m_core first character, suffix family), that is, all manuscript tokens that appear to belong to each grid cell, based on their structural decomposition. These are proxies for the unknown grid contents, not a recovered cipher. If we had the actual keyword and grid, v11 would construct tokens from first principles without needing the manuscript at all.

What v11 provides is therefore a *comparative* argument: does Latin, routed through babuini-type architecture with these proxied cell contents, produce output closer to the manuscript than alternative approaches? The forward cipher is a comparative instrument, not a reconstruction of the historical key.

### 6.2 Results

The forward cipher v11 incorporates four scribe rules: preferential reuse, suffix avoidance, and boundary-biased innovation (all three derived from the copy-mutate production signature documented in Bozzard 2026a), plus column stickiness. The reuse exponent (α = 1.3) was calibrated to match the manuscript's top-50 token concentration (generated 0.411 vs manuscript 0.406); the stickiness parameter (p = 0.22) was calibrated to match the suffix-family bigram rate (generated 0.253 vs manuscript 0.252). Nomenclator-routed function words bypass both rebalancing and stickiness: their family assignment is fixed by the cipher designer, not subject to scribal variation. Both calibration targets are observable properties of the manuscript, not free parameters: the model was required to match them, and the ablation (Supplement S1) confirms that each component improves specific metrics without degrading others. Across ten seeds (42, 404, 501, 606, 808, 909, 101, 202, 303, 505), v11 scores mean 62.9/84 (σ = 2.6), with CORE-15 = 12.6/15 and BG42 = 33.4/42. Zero of 84 metrics are structurally unreachable: every metric passes in at least one seed. The metrics that pass inconsistently cluster in two themes: vocabulary-richness metrics sensitive to the frequency spectrum shape, and word-length distribution moments.

On the Bowern-Gaskell 42-metric benchmark (35 shared metrics), the forward cipher scores 33.4/42, compared with a maximum of 17/35 for any character-level cipher in their 814-text corpus. The word-length standard deviation (v11 σ = 1.88) falls within the syllabic range, well below the character-level floor (σ = 2.37).

An important disambiguation: replacing the inferred nomenclator assignments with random assignments (20 trials, et and in held fixed, remaining 8 words assigned randomly to 6 families) produces scores of 58–66/84 (mean 62.0), overlapping with the inferred assignments' 62.9. The forward cipher score therefore validates the **cipher architecture** (two-table routing, copy-mutate production, calibrated scribe rules), not the specific function-word assignments. The nomenclator's validation comes from the bigram correlation (§5.5): r = 0.96 inferred versus null mean 0.86, p < 0.0001. The architecture and the assignments are validated by different tests.

### 6.3 Cross-section transfer

The nomenclator, derived entirely from Herbal-A analysis and external corpora, transfers to all nine manuscript sections without modification:

| Section | Tokens | Types | Mean n/84 | C15 | BG42 |
|---------|--------|-------|-----------|-----|------|
| Herbal-A | 4,033 | 1,430 | 62.0 | 12.3 | 33.3 |
| Herbal-B | 5,783 | 1,922 | 56.0 | 9.3 | 30.0 |
| Pharmaceutical | 3,870 | 1,599 | 64.7 | 11.3 | 33.3 |
| Rosettes | 1,818 | 797 | 60.3 | 10.0 | 27.7 |
| Stars | 10,702 | 2,982 | 54.3 | 9.3 | 26.0 |
| Zodiac | 1,590 | 873 | 54.7 | 6.3 | 27.0 |
| Cosmological | 1,341 | 809 | 50.7 | 8.0 | 22.3 |
| Astronomical | 1,469 | 775 | 51.3 | 10.3 | 24.0 |
| Balneological | 6,859 | 1,502 | 45.7 | 8.0 | 25.0 |

Eight of nine sections score above 50/84 with zero section-specific tuning and the wrong source text (CI pharmaceutical Latin throughout). Type counts match the manuscript almost exactly in every section, confirming that the vocabulary saturation mechanism generalises. Pharmaceutical and Rosettes score comparably to Herbal-A despite very different content. Stars and Zodiac score mid-range, consistent with a different source genre sharing the same cipher architecture. Balneological scores lowest, consistent with its unusual vocabulary dynamics (high repetition rate, restricted content vocabulary). The architecture generalises; section differences arise from source text variation and scribe differences, not from different cipher mechanisms.

---

## 7. CV Syllable Reader

The nomenclator identifies function words; the grid identifies consonant groups and first vowels. Combining the two produces a CV syllable reader: a tool that converts each VMS token into either a function word class or a consonant-vowel pair. This section tests whether the reader produces meaningful content at the folio level.

### 7.1 Method

Each VMS token is read as follows. EC tokens (empty core) are assigned to their function word family using the inferred nomenclator assignments. FC tokens (filled core) are resolved in two steps: the m_core first character identifies the consonant row (one of seven groups), and the suffix family identifies the first vowel. Within-row consonant resolution uses a frequency-ordering heuristic: within each row, the most common raw core string maps to the most common consonant in the group, the second most common to the second, and so on. For example, in row 'o' (consonants c, s, p), core 'o' (the most frequent) maps to c, cores 'od'/'ol'/'ot'/'ok' (mid-frequency) map to s, and cores 'or'/'octh' (least frequent) map to p.

### 7.2 Roundtrip validation

To verify that the reader resolves what the cipher encodes, we enciphered the Ald.211 Centaurea minor entry (50 Latin words) through the forward cipher, then read the resulting VMS tokens back through the CV reader and compared against the original Latin:

| Level | Accuracy |
|-------|----------|
| Row (7 consonant groups) | 100% (31/31 FC) |
| First vowel (5 suffix families) | 100% (31/31 FC) |
| Individual consonant | 65% (20/31 FC) |

All 11 individual-consonant failures are within-row ambiguity: v/∅ confusion in row 'c' (6 cases), c/s/p confusion in row 'o' (4 cases), d/f confusion in row 'e' (1 case). The row is always correct. The vowel is always correct. The cipher architecture is lossless at the row+vowel level. Within-row disambiguation requires the keyword, which has not been determined. Because the roundtrip uses v11's VMS-derived cell pools, this test validates the internal consistency of the row and family classification scheme, not the historical accuracy of the grid contents.

### 7.3 Directed corroboration test: f2r

Folio 2r is tentatively identified as Centaurea by O'Neill (cyanus silvestris), Holm (Centauria), and Sherwood (Centaurea diffusa), three independent identifiers converging on the same genus through illustration analysis. In pharmaceutical Latin (both CI and Ald.211), Centaurea entries invariably distinguish two varieties: *centaurea maior* (greater centaury) and *centaurea minor* (lesser centaury). The qualifier *minoris/minor* appears repeatedly in every Centaurea minor entry because it is the word that identifies which plant the text discusses. The Ald.211 entry reads: "*centauzee minoris... centauzea mayor et minor...*" The CI entry similarly: "*centaurea maior que am maiorum efficacie et minor qui minorum.*"

The CV syllable 'mi' (m-initial, first vowel i) would be produced by *minoris*, *minor*, *mitigat*, *misce*, *mixtus*, or *mirre*, but of these, only *minoris/minor* is expected to appear at elevated frequency specifically on a Centaurea folio rather than across all herbal folios generally. The others are generic pharmaceutical vocabulary without plant-specific enrichment.

We computed the Ald.211 Centaurea minor entry's distinctive CV profile and ranked all 48 Herbal-A folios by 'mi' enrichment.

Result: f2r ranks **1st out of 48 folios**. 8/46 FC tokens (17.4%) read as 'mi', against a global rate of 1.7%. Hypergeometric p < 0.000001. This survives Bonferroni correction across all folios and CV types (~960 tests). No other folio exceeds 8.7%. [Figure 3 shows f2r's 'mi' rate against the 48-folio distribution.]

The reading adds information to the existing botanical identification: while previous identifiers specified only the genus (Centaurea), the 'mi' enrichment is consistent with *minor* under the model, refining the prior genus-level identification. This refinement is independently testable through morphological comparison of the f2r illustration with known depictions of Centaurea minor versus Centaurea maior. We note that this is a directed test of a known identification, not a blind prediction: the Centaurea identification motivated the search for 'mi'.

Applied systematically to all 48 Herbal-A folios (Supplement S9), five additional folios survive Bonferroni correction across approximately 960 folio-CV tests (threshold p < 0.001). Unlike the f2r result, these enrichments lack independent botanical identifications and therefore constitute CV-level vocabulary constraints, not confirmed content identifications:

| Folio | CV | Count | Rate | Global | Enrichment | p | Candidate Latin |
|-------|-----|-------|------|--------|------------|---|----------------|
| f2r | mi | 8 | 17.4% | 1.7% | 10.4× | 3.7 × 10⁻⁷ | minoris/minor |
| f17v | fe | 4 | 6.2% | 0.6% | 10.7× | 3.2 × 10⁻⁴ | feniculi/febribus |
| f23r | te | 3 | 8.3% | 0.5% | 18.1× | 4.3 × 10⁻⁴ | terantur/tempore |
| f23v | te | 3 | 7.7% | 0.5% | 16.7× | 5.4 × 10⁻⁴ | terantur/tempore |
| f20r | vi | 3 | 6.2% | 0.4% | 15.5× | 6.4 × 10⁻⁴ | vinum/viridis |
| f8v | bo | 2 | 3.8% | 0.1% | 33.4× | 8.8 × 10⁻⁴ | borago/bolus |

The f23r/f23v pair is notable: the same CV enrichment on both sides of the same physical leaf, consistent with a preparation page about grinding or processing (*terantur* = "let them be ground"). f23v additionally shows 'tu' enrichment at p = 0.005 (*tundas* = "pound"). The f17v 'fe' hit was previously tested as a fennel prediction (§9.1) and falsified on illustration grounds; the enrichment likely reflects *febribus* (fevers) rather than *feniculum*, consistent with the same folio's 'de' enrichment (p = 0.002, *decoctio* = decoction). These results do not identify specific plants, but they constrain the pharmaceutical vocabulary on each folio.

### 7.4 Cross-section vocabulary enrichments

The CV reader applied to Herbal-B, a section not used in any part of the reader's construction, produces systematic pharmaceutical vocabulary enrichments:

| Folio | CV | Count | Meaning | p |
|-------|-----|-------|---------|---|
| f27v | cu | × 11 | cutem/cura (skin/cure) | 0.00003 |
| f37v | ci | × 10 | cibis/clister (food/enema) | < 0.00001 |
| f53r | su | × 6 | succum/succo (juice) | 0.00001 |
| f49v | sa | × 11 | sal/sanat (salt/heals) | 0.00022 |
| f56r | sa | × 8 | sal/sanat (salt/heals) | 0.00052 |

These are nominal p-values. Under Bonferroni correction for approximately 1,600 folio-CV tests, f27v, f37v, and f53r survive (p < 0.00003), while f49v and f56r do not.

These enrichments depend on row identification (m_core first character → consonant group) and suffix family assignment, both of which are robust to transcription variation: of the 20 most frequent Herbal-A tokens, none changes its row assignment under known EVA variants (§3.1). The reader was built entirely from Herbal-A analysis and external corpora; no HB data were used in its construction. The presence of systematic pharmaceutical vocabulary enrichments in an unseen section is consistent with the nomenclator and grid capturing genuine cipher properties rather than section-specific artefacts.

### 7.5 Stars section: different genre

The Stars section shows a fundamentally different CV profile from the herbal sections: massively enriched for t-initial and d-initial content words (ta 9.2%, da 8.5% versus HA rates of 1.3% and 2.0%). In Latin, these would be words like *talis*, *tamen*, *tantum*, *dicit*, *dicitur*, *dies*, *dominus*, descriptive and instructional vocabulary rather than pharmaceutical recipe formulas. Ald.211 recipe text matches Stars at only r = 0.31 (versus HA at r = 0.45), which indicates that Stars enciphers a different genre of Latin. The nomenclator (function words) transfers across sections; the content vocabulary does not.

---

## 8. Language Constraints

Four independent tests constrain the source language, with pharmaceutical Latin fitting best. (a) Over 93% of the manuscript's consonant-final tokens end in sonorants (y, n, l, r, or m), excluding Germanic (44–50%), Semitic (60–65%), and leaving only Romance languages. (b) Under grid assignments, the FC consonant distribution fits Circa Instans pharmaceutical Latin at χ² = 0.04, Italian vernacular at χ² = 0.13, and all non-Romance languages at χ² > 0.5. Genre discrimination is strong: non-pharmaceutical Latin fits 6–10× worse. (c) bare-Y = *et* is specifically Latin (Italian uses *e*). (d) The 53/47 EC/FC ratio matches Latin pharmaceutical prose. These constraints were established independently of the assignment inference and reinforce its pharmaceutical Latin assumption. Bozzard (2026a) demonstrated that the PGCS grammar accommodates every language tested as hypothetical plaintext; the constraints reported here operate at a different level, not the grammar's slot rules (which are language-agnostic), but the bigram sequences and consonant distributions within the grid cells (which are language-specific).

---

## 9. Negative Results

The assignment inference and CV reader succeeded at the levels reported above. This section documents the hypotheses we tested and rejected, because the method's limits are as informative as its successes.

### 9.1 f17v fennel corroboration test (false positive)

The CI fennel entry's CV profile correlates most strongly with f17v out of 48 HA folios (r = 0.365, rank 1/48, null model p = 0.0008 from 5,000 random CI passages of the same length). However, f17v's illustration depicts a broad-leaved plant with berry clusters, tentatively identified as Tamus communis (Zandbergen: ELV tamus, ThP smilax/tamus communis). This is unambiguously not fennel (feathery leaves, yellow umbels). The statistical correlation likely reflects fennel as a listed ingredient in the plant's pharmaceutical preparations, or the presence of fe-initial vocabulary (*febribus*, *fel*) unrelated to the plant name. This establishes the method's principal limit: CV profile matching produces false positives when vocabulary overlap is driven by shared ingredients or shared therapeutic vocabulary rather than plant identity.

### 9.2 Two-syllable hypothesis (killed)

We tested whether consecutive FC tokens encode consecutive syllables of the same word. VMS FC doublets between EC anchors match CI two-syllable words at 45.6%, versus null model rates of 47.5% (random CV pairs, p = 0.73) and 49.8% (shuffled VMS FC stream, p = 0.88). The match rate is below chance. Each FC token encodes one word's first CV independently; consecutive tokens are independent words, not consecutive syllables.

### 9.3 Nomenclator extension (ceiling confirmed)

We tested every unassigned EC word with frequency ≥ 3 in Ald.211 individually against the optimiser. No genuine Latin function word improves when reassigned from its vowel-heuristic family. Content words (folia, splenis, epatis, puluis) produce marginal improvements (Δr = 0.001–0.004) but represent overfitting to the training corpus's specific botanical vocabulary: these words would not generalise to a different pharmaceutical text. The 10-word nomenclator is the inferable ceiling using this method.

### 9.4 Additional killed hypotheses

| Hypothesis | How tested | Why rejected |
|-----------|------------|--------------|
| CI is the source text | Enciphered CI, compared individual consonants | 1/11 rows match |
| Per-section keywords | 278 keywords through forward cipher | All score 51–53/84 |
| Ring model for EC equivalences | Transitivity test | 0.31–0.38, far below 1.0 |
| Grabadin vocabulary mapping | Non-circular assignment test | 0/5 matches |
| Track B ecological controls on v11 | Three hybrid engine attempts | All scored below both parents |
| Gallows-aware generation (v12) | Added gallows axis to forward cipher | 66–69, no improvement |
| Copy-mutate on character-level output | Applied scribe rules to simple cipher | σ remains ≈ 2.4 |
| Verbose homophonic without grid (Naibbe-type) | Greshko (2025) §4.2–4.3 | Lacks CV-syllable structure and long-range correlations |

---

## 10. What Remains Open

Amadi's babuini system (§§0074-0078) specifies three things the cipher designer must set: the grid itself (which syllable fills each cell), the keyword (which permutes consonant-row assignments), and the nomenclator table (which function words map to which houses). In classical cryptanalysis, "recovering the key" means recovering all three. For the VMS, they represent distinct recovery problems with different difficulty.

**The grid cell contents.** We know which row a Latin initial consonant maps to, and which column (suffix family) a Latin first vowel selects. We do not know which specific core string the grid designer placed in each cell. The forward cipher proxies this with observed VMS tokens grouped by row and family, but these pools contain every token ever used in that cell, not the single string the designer specified. Recovering the cell contents requires either an external key document or enough known-plaintext cribs to read individual cells directly. Exhaustive row-permutation search (5! = 120, with two rows fixed by confirmed mappings) finds that the identity permutation produces the best fit for Herbal-A. If a keyword exists, it is trivial for this section: the grid appears to be in its natural consonant order. This does not mean no keyword was used; it means the keyword's effect on row assignments is either null or equivalent to the identity mapping for the section analysed.

**The nomenclator house assignments.** The EC cross-validation (§5.5) constrains which function words belong to which suffix family, but not which of the four houses within a family each word occupies. Amadi describes house selection as polyalphabetic (rotating by position or entry), but the specific rotation rule for the VMS has not been determined. Gallows distribution (§4.3) suggests house assignment correlates with document structure (p/f at entry boundaries, k within entries, t at sub-entries), but the mapping from house to gallows character is not yet recovered.

**The source text.** Pharmaceutical Latin in the CI tradition fits the consonant distribution and function-word profile, but the specific manuscript being enciphered has not been identified. CI itself is excluded (§9); the source is something CI-derived. The two autocorrelation metrics that never pass in the forward cipher (0/10 seeds) measure where vocabulary clusters in the token stream — topical structure that depends entirely on which plant entry is being enciphered in which order. No amount of key recovery fixes this; it requires identifying the actual book. The Stars section, with its distinct vocabulary profile (§7.5), may require a different genre source entirely.

---

## 11. Conclusion

We have proposed a two-table cipher architecture for Beinecke MS 408, a function-word nomenclator paired with a content-word syllabic grid, and tested it against independent pharmaceutical Latin corpora. Using manuscript-constrained optimisation on external Latin text, we inferred candidate suffix-family assignments for ten function words, cross-validated at r = 0.89, exceeding all 10,000 random assignments. A directed corroboration test on folio 2r (Centaurea minor, p < 10⁻⁶) and five additional Bonferroni-surviving CV enrichments are consistent with pharmaceutical vocabulary under this model.

The copy-mutate scribal production layer (Bozzard 2026a) explains why the nomenclator was not previously detectable: the scribe's variation destroys the one-to-one mapping between cipher output and surface tokens that classical frequency analysis requires. Operating at the suffix-family level obviates this problem, because family identity is preserved through copy-mutate variation even as the surface form changes.

Davis (2020a) sets three criteria for an acceptable Voynich proposal: it must be consistent with the object, follow a sound and repeatable methodology, and produce a reading that makes sense. Bowern and Lindemann (2021) add that any proposed decipherment "must work consistently across the text as a whole." The present work meets the first two Davis criteria and the Bowern-Lindemann cross-corpus test: the architecture is consistent with the object's radiocarbon date, Northern Italian provenance, and codicological structure; the methodology is fully reproducible from archived code and data; and the cipher-class identification applies across the whole corpus without modification, evaluated on all nine manuscript sections and all 224 folios. Davis's third criterion — a reading that makes sense — remains unmet. The grid cell contents, complete nomenclator house assignments, and source text are unknown. Without them, the architecture produces family-level constraints (this token encodes a c-initial, e-vowel Latin word) but not readings (this token encodes *centaurea*). The distance between constraint and reading is not a single keyword in the classical sense, but the complete specification of the cipher tables — what Amadi calls the grid, the houses, and the nomenclator — together with the source text being enciphered. What we have demonstrated is the identification of a constrained cipher architecture, a candidate function-word family mapping under external pharmaceutical Latin, and a set of folio-level vocabulary constraints surviving multiple-testing correction. The architecture is assembled from components individually attested in fifteenth-century Northern Italian cryptographic practice; the full integrated form is directly documented later in Amadi's treatise.

All data and code are available at DOI 10.5281/zenodo.18812705.

---

## Acknowledgements

We thank D.P.J.A. Scheers for his edition of Amadi's treatise, without which the systematic review of Venetian cipher practice would not have been possible. We thank Marco Ponzi for correspondence on Amadi's alphabet reduction. The Circa Instans transcription was produced from Wellcome Collection MS 624 using Transkribus Text Titan I ter (model ID 356425). We thank René Zandbergen for maintaining the essential voynich.nu resource. We are grateful to Michael Greshko for the Naibbe cipher, which clarified both the strengths and gaps of verbose homophonic approaches. We thank the Voynich Ninja forum for critical engagement, and the Beinecke Rare Book & Manuscript Library for high-resolution manuscript images.

Computational analysis was performed with Claude (Anthropic). The author takes full responsibility for all claims, errors, and interpretations.

---

## Figures

**Figure 1.** EC-EC bigram correlation before and after nomenclator assignment. Left: vowel heuristic only (r = 0.31). Right: with ten inferred assignments (r = 0.96). Each point is one of 43 non-zero bigram types.

**Figure 2.** Leave-one-out analysis. Δ training r when each word is removed. *et* accounts for 76% of the total improvement; *in* contributes zero.

**Figure 3.** CV syllable 'mi' enrichment across 48 Herbal-A folios. f2r (17.4%) is a clear outlier against the global rate (1.7%, dashed line).

**Figure 4.** Forward cipher cross-section scores (mean n/84, three seeds per section). All nine sections scored without section-specific tuning, using CI source text throughout. Eight of nine exceed 50/84.

---

## References

Bowern, C. L., and Gaskell, D. E. (2022). Enciphered after all? Word-level text metrics are compatible with some types of encipherment. ICVM 2022, University of Malta.

Bowern, C. L., and Lindemann, L. (2021). The Linguistics of the Voynich Manuscript. *Annual Review of Linguistics*, 7(1), 285–308.

Bozzard, E. (2026a). A four-slot grammar confirmed from within the Voynich Manuscript: structure, production evidence, and the limits of decipherment. Submitted to *Cryptologia*.

Cheshire, G. (2019). The language and writing system of MS408 (Voynich) explained. *Romance Studies*, 37(3), 131–167.

Davis, L. F. (2020a). What will it take to solve the Voynich Manuscript? *Manuscripts*, 72(2), 73–85.

Davis, L. F. (2020b). How many glyphs and how many scribes? *Manuscript Studies*, 5(1), 164–180.

Greshko, M. A. (2025). The Naibbe cipher. *Cryptologia*. DOI: 10.1080/01611194.2025.2566408.

Kahn, D. (1967). *The Codebreakers*. Macmillan.

Meister, A. (1902). *Die Anfänge der modernen diplomatischen Geheimschrift*. Schöningh.

Pelling, N. (2016). Fifteenth century cryptography. *Cipher Mysteries*.

Scheers, D. P. J. A. (2020). *Giovanni Battista Amadi: Trattato delle cifre*. Self-published.

Strong, L. C. (1945). Anthony Askham, the author of the Voynich Manuscript. *Science*, 101(2633), 608–609.

Tucker, A. O., and Talbert, R. H. (2014). A preliminary analysis of the botany, zoology, and mineralogy of the Voynich Manuscript. *HerbalGram*, 100, 70–85.

Zattera, L. (2022). A slot grammar for the Voynich Manuscript. *Cryptologia*, 46(5), 411–436.

---

## Supplements

- **S1:** Forward cipher v11 with nomenclator, column stickiness, and calibrated reuse (v11_nomenclator.py)
- **S2:** Nomenclator optimizer (nomenclator_optimizer.py) - standalone inference pipeline, reproduces all results from two input files
- **S3:** CV syllable reader (cv_folio_reader.py) - reads any folio, runs enrichment test
- **S4:** Forward cipher v11 clean baseline (forward_cipher_v11_CLEAN.py)
- **S5:** 84-metric scoring battery (score_85_metrics.py)
- **S6:** Language discrimination battery (16 candidates, 4 tests)
- **S7:** EVA robustness tests (ZL/ZI transcription variants, 0/20 top tokens affected)
- **S8:** Leave-one-out analysis (full table with training, CI, and held-out metrics)
- **S9:** Folio enrichment results (complete table, HA and HB, all CVs)
- **S10:** Negative results: f17v false positive analysis, two-syllable null model, nomenclator extension test, hybrid engine attempts
- **S11:** BG42 comparison data (v11 vs 814 character-level ciphers)
- **S12:** Killed hypotheses with test data and failure diagnoses
