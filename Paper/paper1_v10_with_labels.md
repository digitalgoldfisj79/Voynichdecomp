# Slot Grammar and Self-Citation in the Voynich Manuscript: A Generator Hierarchy

Edward Bozzard

*Submitted to Cryptologia, February 2026*

## Abstract

Gaskell and Bowern (2022) proposed that future work on the Voynich Manuscript should "construct automated algorithms which can generate larger volumes of gibberish" to test whether VMS-like text can be produced without linguistic meaning. This paper implements that programme.

We formalise the four-slot PGCS grammar (Prefix + Gallows + Core + Suffix) that governs 92.96% of the manuscript's 37,465 tokens and decompose word selection into a grammar layer (28.9% of word-selection entropy) and a content layer (71.1%). Attested slot co-occurrences compress the generative space 127-fold below the free combinatorial product, while character-level constraints over-generate by at least 700×, quantifying the grammar's empirical bite. We then construct a six-tier generator hierarchy implementing self-citation with increasingly specified structural ledgers, scoring each tier against 90 distributional metrics extracted from the corpus.

Without slot grammar, generators reproduce 37 to 50 of 90 metrics. With grammar, they reach 58 to 67, a categorical jump confirming grammar as both necessary and sufficient to reach the grammar-constrained statistical regime. The residual gap concentrates in precisely those metrics Gaskell and Bowern (2022) independently identified as the strongest discriminators between gibberish and meaningful text: word-length autocorrelation (AC(1) = +0.160), lexical repetition rate (0.008), and local vocabulary diversity (MATTR-25 = 0.919). This convergence is not an artefact of our method; the metrics were selected by a random forest classifier and by our generator hierarchy independently. However, the VMS clusters with gibberish rather than natural language on these metrics (Gaskell and Bowern 2022), so the residual gap characterises the production mechanism without resolving whether it encodes semantic content.

A bimodal vocabulary structure provides an additional constraint: 68% of word types are hapax legomena with filled morphological cores (97%), while the remaining 32% of types account for 86% of running text, of which 61% consists of empty-core tokens. This structural split does not emerge from self-citation or from any generator we tested, and requires explanation under any production hypothesis.

Self-citation is a real component of VMS text production, but it operates within a grammar more constrained than self-citation alone generates, shared across multiple scribal hands, and producing two structurally distinct word classes whose origin remains an open question.

**Keywords:** Voynich Manuscript, self-citation, slot grammar, generator hierarchy, computational linguistics, production mechanism

## 1. Introduction

### 1.1 The Paradox

The Voynich Manuscript has resisted decipherment for over five centuries (d'Imperio 1978). Yet it is not random. Character-level bigram entropy (h₂ ≈ 2.1 bits) falls within the range of natural scripts. Word-frequency distributions follow Zipf's law (R² = 0.915; Zipf 1949). Vocabulary diversity matches natural language corpora, and five manuscript sections (herbal, pharmaceutical, biological, astronomical, and recipe) are distinguishable by vocabulary alone at 76 to 81% accuracy (Montemurro and Zanette 2013; Bowern and Lindemann 2021). By every surface metric, the text behaves like language.

The paradox is that beneath this surface, every attempt to recover lexical meaning has failed. No cipher proposal has produced readable text. No statistical clustering of tokens aligns with the manuscript's illustrations. Shuffling word order within folios does not degrade section classification, because the signal is purely vocabulary-based, not syntactic. The text has structure without recoverable semantics.

### 1.2 Three Traditions

Three lines of argument dominate VMS scholarship, and recent work has sharpened each into a testable position.

The cipher tradition assumes the text encodes natural language. All published decipherments have failed reproducibility tests (Newbold 1928; Strong 1945; d'Imperio 1978; Reddy and Knight 2011). Most recently, Greshko (2025) demonstrated a historically plausible verbose homophonic substitution cipher — the Naibbe cipher — that encrypts Latin and Italian into ciphertext replicating many VMS word-level statistics simultaneously. Greshko's cipher uses an expanded version of the Zattera slot grammar as its structural backbone and produces ciphertexts that outperform 86% of texts in Bowern's 932-document corpus at matching VMS. His random forest classifier rates these ciphertexts as "gibberish" at low confidence, exactly as it rates VMS. The cipher does not claim to be the VMS solution; it demonstrates that the ciphertext hypothesis remains viable. It also reproduces only approximately 30% of unique Voynich B word types.

The hoax tradition proposes that the text was generated mechanically. Rugg (2004) demonstrated Cardan grille production; Schinner (2007) argued on statistical grounds. Timm and Schinner (2020) proposed a concrete mechanism that unifies the stochastic and hoax traditions: self-citation, the iterative copying and modification of previously written words. Their algorithm produces Zipf-compliant text with VMS-like frequency distributions. Timm and Schinner (2024) extended this argument, and Timm (2026) makes the strongest claim yet: self-citation explains the text fully, and static analytical frameworks are invalid because the text is inherently dynamic.

The linguistic tradition documents statistical regularities without claiming decipherment. Currier (1976) and Tiltman (1967) identified positional character preferences. Matlach, Janečková, and Dostál (2022) found symbol-role patterns using information-theoretic analysis. Most consequentially, Gaskell and Bowern (2022) conducted the first experimental study of VMS-like text production, recruiting 42 volunteers to write meaningless text and comparing the results against VMS and a corpus of meaningful texts. Their random forest classifier identified VMS as more closely resembling gibberish than meaningful text, with the strongest discriminating features being character-position bias, word-position bias, compression, repeated words, and word-length autocorrelation.

Gaskell and Bowern (2022) identified one metric their gibberish samples could not replicate: the VMS's unusually large bias in character placement within words (charbias_words_mean). They attributed this to "typographic considerations which cannot be tested rigorously using texts restricted to the lowercase Latin alphabet." This metric corresponds to what we formalise below as the PGCS slot grammar.

### 1.3 This Paper

Gaskell and Bowern (2022) concluded that "a viable approach to future work... may be to use the lessons learned here to construct automated algorithms which can generate larger volumes of gibberish." This paper implements that programme, with three extensions.

First, we formalise the positional character constraints as a four-slot morphological grammar (PGCS) and validate it through a six-test falsification protocol. This addresses the sole VMS metric Gaskell and Bowern's gibberish could not replicate.

Second, we build a six-tier generator hierarchy that implements self-citation within progressively more detailed structural constraints, quantifying exactly which VMS properties arise from which production mechanisms. This tests Timm and Schinner's (2020) self-citation hypothesis directly and systematically.

Third, we characterise a bimodal vocabulary structure — hapax legomena with filled morphological cores versus high-frequency repeated types with empty cores dominating running text — that constrains any proposed production mechanism and is not reported in previous analyses.

Section 2 formalises the PGCS slot grammar. Section 3 quantifies the grammar/content boundary through an information budget. Section 4 builds the generator hierarchy and identifies the residual gap. Section 5 discusses the results in direct conversation with Timm (2026), Gaskell and Bowern (2022), and Greshko (2025). Full generator specifications, metric definitions, and scoring code are available at [GitHub/Zenodo DOI].

## 2. The PGCS Decomposition

### 2.1 Corpus

All analyses use the Zandbergen-Landini-Zandbergen-IVTFF (ZLZI) transcription of the Voynich Manuscript (Zandbergen 2024), comprising 37,465 tokens and 7,598 types across 226 folios. This transcription system uses Extended Voynich Alphabet (EVA) characters. Sensitivity analysis against six transcription systems and under 5% character corruption (Supplement S4) confirms that structural findings are robust to transcription-level variation, with MI values shifting by ±0.05 bits.

### 2.2 The Four-Slot Model

The Prefix-Gallows-Core-Suffix (PGCS) model decomposes each token into four ordered slots.

The Prefix slot (8 types) occupies word-initial position before any gallows character. The critical innovation of PGCS is the reassignment of *ch* and *sh* from gallows to prefix status. These characters behave distributionally as prefixes: they appear at word-initial position, combine freely with following gallows characters, and affect 9,134 tokens directly (24.4%); the cascading effect on suffix boundaries means that 24,964 tokens (66.6%) parse differently under PGCS than under any three-slot alternative. Under previous three-slot models, *ch* and *sh* were grouped with gallows characters, obscuring the true slot structure.

The Gallows slot (9 types) contains the tall characters *k*, *t*, *p*, *f* and their bench variants *ckh*, *cth*, *cph*, *cfh*, plus the rare *m*. The Core slot (open class) carries the central morphemic material; critically, 52.7% of tokens have empty cores. The Suffix slot (33 types) is a closed inventory of terminal material with highly regular distributional properties.

The reclassification of *ch* and *sh* from gallows to prefix is the most consequential boundary decision, affecting 24,964 tokens. It produces the 52.7% empty-core rate, nearly halves core-suffix mutual information (from 1.860 to 0.976 bits), and recasts common words like "chedy" as function words [ch | ∅ | ∅ | edy] rather than content words [∅ | ch | e | dy]. The four-slot design is further motivated by the scale difference between gallows (9 types) and core (2,001 types), by the prefix-gallows association being the strongest pairwise coupling in the system (Cramér's V = 0.266), and by the fact that the empty-core rate is invisible under any model that folds gallows into core.

### 2.3 Validation

We formalise the PGCS grammar as 210 rules: 81 character-sequence constraints, 52 pair-adjacency rules, 41 suffix rules, and 36 prefix rules. The grammar derivation methodology, including rule discovery from character co-occurrence patterns, the data-driven reclassification of *ch* and *sh*, and the train/test protocol with 40% held-out data frozen at the P69 stage, is documented in Supplement S1. The merged character grammar achieves character coverage of 92.96% and fully parses 71.93% of word types (99.87% partial coverage). The 28.07% of types not fully parsed contain rare character bigrams within their cores that fall outside the 81 character-sequence rules, but every attested token receives a valid four-slot decomposition regardless. The uncovered characters concentrate in low-frequency types and do not form productive patterns of their own.

To test generalisation, we held out 40% of the corpus by folio. The held-out set contains 3,930 unique types not seen during rule development; zero produce PGCS violations. The grammar generalises perfectly to unseen data. Position-conditioned quintuples (prefix, gallows, core-class, suffix-family, line-position) provide a further test: 6,750 observed quintuples from 37,465 tokens represent a 292-fold compression from the theoretical combinatorial space, again with zero validation failures.

A further out-of-domain test applies the grammar to labels, the short tokens attached to illustrations throughout the manuscript, structurally distinct from running paragraph text. Of 699 Voynichese label types (975 tokens across 835 label lines), every type receives a valid PGCS parse: zero violations. The grammar generalises perfectly to a text mode it was not developed on. However, labels show a markedly different slot-filling distribution. The o-prefix accounts for 52.1% of label tokens versus 20.5% in paragraph text (2.54× enrichment), while qo-prefix nearly vanishes (0.8% vs 14.9%). Filled-core tokens constitute 70.9% of label tokens versus 45.8% in paragraphs, and 51.4% of label types appear nowhere in running text, of which 93.6% are corpus hapax. Labels draw from the hapax-generating stratum rather than the copy-mutate stratum identified in §4.2. The PGCS architecture is shared; the production parameters are not. This is consistent with format-driven production in which labels receive individually generated tokens because pharmaceutical manuscript conventions require distinct identifiers, but the mirror property (§5) means the grammar cannot distinguish semantic labels from generated ones. The label question remains underdetermined by structural evidence.

Comparison against all alternative slot assignments yields a decomposition error of 0.001 bits for PGCS versus 1.074 bits for the next-best alternative, a roughly 1,000-fold difference. The most competitive alternatives either fold *ch*/*sh* back into gallows (producing a 3-slot model that masks the empty-core rate and inflates core-suffix MI to 1.860 bits) or merge gallows into core (producing a 3-slot model where the core slot spans 9 to 2,001 types, collapsing the cardinality distinction that separates closed-class from open-class behavior). Both alternatives degrade held-out generalisation and produce higher redundancy between slots. Adversarial tests targeting circularity, overfitting, and transcription dependence (Supplement S4) likewise fail to produce degradation, supporting the interpretation that the slot architecture reflects underlying structure rather than model-specific fitting.

The PGCS grammar corresponds to what Gaskell and Bowern (2022) measured as charbias_words_mean — the sole metric their experimental gibberish could not replicate. What they attributed to "typographic considerations" turns out to be morphological: a systematic four-slot grammar with 210 falsifiable rules, not a visual artefact of scribal aesthetics.

## 3. The Grammar-Content Boundary

Word-selection entropy partitions into a grammatical component and a content component. The partition emerges from three axes of analysis: slot-level structure within words, sequential and positional constraints across words, and their combination into an information budget.

### 3.1 Within-Word Structure

The chain rule of entropy (Shannon 1948; Cover and Thomas 2006) guarantees H(word) = H(P) + H(G|P) + H(C|P,G) + H(S|P,G,C) for any lossless decomposition. The empirical finding lies in the gap between this sum and the sum of marginal slot entropies. Unconditional slot entropies sum to 13.171 bits, exceeding H(word) = 10.311 by 2.860 bits (21.7% redundancy). PGCS slots are approximately 78% independent, with the remaining 22% carried primarily by the core-suffix association (MI = 0.976 bits, full 2,001-type core), followed by prefix-core (MI = 0.428 bits) and prefix-gallows (MI = 0.393 bits).

More than half of all tokens (52.7%) have empty cores, composed entirely from closed-class inventory items. This rate varies by section, from 37.7% (Cosmological) to 63.5% (Balneological). Unlike natural language, where content words dominate running text, this manuscript's vocabulary is majority-functional. Second-order character entropy (h₂ = 2.13 bits) nonetheless falls within the natural language range documented by Bowern and Lindemann (2021), ranging from 1.96 (biological) to 2.23 (pharmaceutical) across sections.

### 3.2 Across-Word Structure

If words were selected independently, even from position-specific pools, adjacent word lengths would show negligible correlation. In fact they are positively correlated: AC(1) = +0.160. We prove (Supplement S5.4) that under independent sampling the expected autocorrelation magnitude is bounded by approximately 0.025 given VMS parameters. The observed value exceeds this bound by more than sixfold, and simulation with 10,000 replications confirms that no replication exceeds |AC| = 0.028. Independent word generation is formally excluded as a production mechanism.

Gaskell and Bowern (2022) identified positive word-length autocorrelation as a distinctive property shared by VMS and their gibberish samples, separating both from natural language (where AC is typically negative or near-zero). The positive AC is therefore not evidence of hidden information but a production-mechanism signature. The structural pathway runs through PGCS: suffix-family constrains the following prefix, and prefix constrains word length, so the coupling propagates through the slot system rather than through direct length dependence. The effect resets at line boundaries, consistent with within-line AC (0.151) exceeding cross-line AC (0.062).

The suffix-to-prefix transition grammar yields an 8×8 transition matrix with strong directional biases: Y-suffix → qo-prefix (26.0%, 1.9× enrichment), BARE-suffix → ∅-prefix (39.5%, 1.8×), N-suffix → o-prefix (29.7%), and R-suffix → o-prefix (26.1%). Line-initial tokens show a distinct distribution with elevated y-prefix (13.4%), s-prefix (9.3%), and d-prefix (14.3%). This transition grammar contributes MI(quad; prev_sfx) = 0.757 bits beyond section and position conditioning combined, the single largest additional predictive axis.

The positional grammar confirms Currier's (1976) observation that the line is a functional unit. Word selection correlates with line position at MI(quad; position) = 0.380 bits, and line structure follows a three-zone pattern. The opener zone is enriched for d-prefix (15.3% vs 10.1%, Z = 12.5) and depleted for empty cores (39% vs 57%). The closer zone carries the M-suffix line-ending marker at 14.9% versus 1.8% at the penultimate position (Z = 57.8), by far the strongest positional signal in the manuscript. First-line-of-paragraph openers (226 tokens) are particularly striking: 71.7% carry ∅-prefix, 84.5% bear gallows, and only 12.4% have empty cores. This paragraph-marker class has no known analogue in published analyses of the Voynich Manuscript.

### 3.3 The Information Budget

The preceding analyses identify five conditioning axes (full derivation in Supplement S2). We compute cumulative mutual information between each axis and the PGCS quad (prefix, gallows, core-class, suffix-family), adding axes in decreasing order of marginal contribution:

| Conditioning (cumulative) | MI (bits) | % of H(quad) | Δ added |
|--------------------------|-----------|--------------|---------|
| + Section | 0.810 | 8.9% | 0.810 |
| + Line position | 1.329 | 14.6% | 0.518 |
| + Previous suffix family | 2.086 | 22.9% | 0.757 |
| + Paragraph flag | 2.116 | 23.2% | 0.030 |
| + Quire (production unit) | 2.634 | 28.9% | 0.518 |
| **Total explained** | **2.634** | **28.9%** | |
| **Unexplained (lexicon)** | **6.490** | **71.1%** | |

*Table 1. Information budget. Five conditioning axes explain 28.9% of quad entropy (H(quad) = 9.124 bits). The remaining 71.1% is the lexical content layer.*

The largest single increment comes from the sequential transition axis (0.757 bits), identifying suffix-to-prefix coupling as the dominant sequential structure. The paragraph flag contributes negligibly (0.030 bits) despite its striking distributional signature, because the 226 paragraph-initial tokens represent only 0.6% of the corpus. Quire identity adds 0.518 bits beyond section, reflecting within-section vocabulary variation across production units, consistent with cross-quire lexical clustering documented independently through paleographic analysis (Fagin Davis 2020). Because quire and section are substantially confounded (NMI = 0.851, with only 4 of 16 quires spanning multiple sections), this increment is computed conditionally on all prior axes and captures variation not already explained by section, position, or sequential context; supporting evidence includes the cross-quire lexical cluster (f42–f49–f56, Jaccard = 0.108, 98 exclusive quads), which documents genuine within-section quire-level vocabulary differentiation.

The 71.1% unexplained entropy is the content signal: the specific lexical choices that distinguish one passage from another within the same section, position, and grammatical context. "Content" here denotes unexplained statistical variability in word selection, not recoverable semantic meaning; the information budget is agnostic about whether this residual encodes language, carries structured nonsense, or reflects some intermediate state. The grammar (29%) is fully reproducible without semantic encoding. The content (71%) is what resists recovery. The remainder of this paper tests how far self-citation can reach into that content layer.

## 4. Self-Citation and Beyond

### 4.1 Structural Complexity of the PGCS Lexicon

The grammar produces structured text. Before we can test how far self-citation reaches, we need to establish what any production mechanism must reproduce.

The strongest coupling in the dataset is between core and suffix-family selection (MI = 0.474 bits; Z = 215.0 against the null of independence): certain cores preferentially take certain suffixes, so VMS words are not randomly assembled. This coupling interacts with position. The 142 cores occurring ten or more times partition into opener-biased, closer-biased, and position-neutral classes (χ² = 2608.6, p ≈ 0), and even visually similar cores (edit distance = 1) produce significantly different suffix profiles in 73.6% of 280 tested pairs. The grammar tracks morphemic identity at fine granularity.

At the vocabulary level, the lexicon stratifies into functional and content-bearing components. A set of whole-word tokens behave as function-word candidates, with coefficient of variation below 0.37 across sections and typically complex PGCS structure with no core (a full list is available in the repository). At the other extreme, a comparable number of cores concentrate in one or two sections (CV > 0.8), recapitulating the vocabulary differentiation that drives section classification but localising it to the core slot. The Currier A/B "languages" similarly manifest as differential word selection within a shared grammar rather than as distinct grammars.[^currier-bimodal]

The suffix decomposition reveals a further structural property. The coupling between vowel prefix and terminal consonant within suffixes is strong (I(VP; terminal) = 1.478 bits; NMI = 0.644), meaning that not all vowel-prefix/terminal combinations are equally available. The suffix operates as a constrained combinatorial system rather than a free product of independent axes. A stochastic generator implementing the PGCS slot architecture reproduces this coupling (NMI = 0.615), while generators lacking slot structure produce substantially weaker coupling (NMI = 0.403 to 0.473). This bounds the minimum structural complexity required to produce VMS-like text.

These coupling and stratification findings can be made precise through an over-generation test. A production mechanism's quality depends as much on what it excludes as on what it produces. We define the over-generation ratio as the number of distinct token forms a mechanism licenses divided by the number attested in the manuscript (7,598 types).

Three levels of constraint yield sharply different ratios. First, a character-adjacency ledger recording which characters may follow which in positional context — 584 transitions, the level of constraint described by recent ledger-based production models. Walking this ledger generates at least 5.3 million legal token forms (a lower bound; exhaustive enumeration at lengths 1–6 alone yields 3.88 million, with exponential growth at each additional character). Precision is 0.07%: fewer than one in a thousand legal walks produces an attested VMS token. The ledger cannot reach 48% of attested types at all, including high-frequency tokens such as *qokeedy* (306 occurrences) and *qokaiin* (265 occurrences). Character-level adjacency constraints over-generate by at least 700×.

Second, the free product of PGCS slot inventories (8 prefixes × 9 gallows × 1,302 classified core types × 7 suffix families) yields 656,208 theoretical quadruples, an 86× over-generation ratio. The 1,302 classified cores collapse the 2,001 raw core strings (§2.2) by grouping surface variants that share the same slot behaviour; the free product at either granularity vastly exceeds the attested vocabulary. Without co-occurrence constraints, the slot architecture still admits forms vastly outnumbering the attested vocabulary. This is the criticism raised by Stolfi (2005): a decomposition that permits all slot combinations explains nothing about which combinations actually occur.

Third, the attested slot co-occurrences resolve both problems. Only 5,172 of 656,208 possible quadruples (0.79%) are observed in the manuscript — a 127-fold compression from the free product. Of these, 75.4% have exactly one surface realisation: for three-quarters of the PGCS vocabulary, the quadruple uniquely determines the token. When line position is added as a fifth conditioning axis, 6,750 quintuples are attested from a theoretical space of 1,968,624, a 292-fold compression.

| Constraint level | Licensed forms | Over-generation ratio |
|-----------------|----------------|----------------------|
| Character-adjacency ledger | ≥5,323,194 | ≥701× |
| Free PGCS (P × G × C × S) | 656,208 | 86× |
| Attested PGCS quadruples | 5,172 | 0.68× |
| Attested PGCS quintuples | 6,750 | 0.89× |

*Table 2. Over-generation hierarchy. Character-level constraints produce at least 700× more forms than attested. Unconstrained PGCS reduces this to 86×. Attested slot co-occurrences compress the space 127-fold from the free product, producing a ratio below 1.0 because multiple surface types map to the same quadruple (mean 1.47 surface forms per quadruple). The grammar does not over-generate; it under-specifies only at the level of surface realisation within validated slot combinations.*

This compression ratio is the empirical measure of the work performed by slot co-occurrence constraints. The sequential transition grammar (§3.2) provides a further probabilistic constraint: while all suffix-to-prefix transitions are technically attested, their probabilities are highly non-uniform (Y-suffix → qo-prefix at 1.9× enrichment, BARE → ∅-prefix at 1.8×), contributing 0.757 bits of mutual information beyond what section and position conditioning explain.

### 4.2 The Bimodal Vocabulary

A structural property not previously reported in VMS scholarship constrains the production mechanism further, and it emerged not from a planned test but from diagnosing why our generators persistently underproduced hapax legomena. The manuscript's vocabulary is sharply bimodal.

Of the 7,598 word types, 5,189 (68.3%) are hapax legomena — types occurring exactly once. The remaining 2,409 types (31.7%) account for 32,276 tokens (86.2% of running text). These two classes differ systematically in morphological structure. Among hapax types, 97% have filled cores (non-empty core slots). Among non-hapax types, 82% have filled cores — but the 18% with empty cores are disproportionately high-frequency, so that 61% of non-hapax running text consists of empty-core tokens. The top ten empty-core types alone (*daiin*, *ol*, *chedy*, *aiin*, *shedy*, *chol*, *ar*, *or*, *chey*, *s*) account for 4,858 tokens. The average frequency of repeated types is approximately 13.4 occurrences.

The hapax types arise from two sources. The majority (63.3%, or 3,285 types) represent unique PGCS quadruples — slot combinations that appear nowhere else in the manuscript. The remainder (36.7%, or 1,904 types) are surface variants of quadruples that also produce other forms. Of all observed quadruples, 75.4% have only one surface realisation. The core slot drives this diversity: 783 classified core-classes appear exactly once, and 1,029 (79% of all distinct core-classes) appear three times or fewer.

Timm and Schinner's (2020) self-citation model predicts that hapax legomena should be accidental products of the copy-modify process — words that happened not to be reused. Under that model, hapax and non-hapax types should have the same internal morphological structure, differing only in how often they were copied. They do not. The 97% filled-core rate among hapax types versus 61% empty-core rate in non-hapax running text (token-weighted) is not a copy failure; it is a structural distinction in the production mechanism. Whatever process generated the manuscript's text used one strategy for producing high-frequency template words (empty-core, closed-class) and a different strategy for producing the long tail of single-use types (filled-core, open-class).

This bimodal structure connects to the information budget. The 71.1% unexplained entropy — the content layer — is concentrated in the hapax stratum. Generators can match the function-word layer through copy-modify operations. They cannot match the hapax layer because it requires making different lexical choices in different contexts, precisely the variability that the content layer measures.

### 4.3 Mechanism Exclusions

The information budget and structural complexity documented above formally exclude several classes of production mechanism (full derivations in Supplement S5). Each exclusion eliminates a region of mechanism space by identifying a specific empirical constraint that the candidate process cannot satisfy.

Monoalphabetic substitution over natural-language prose is excluded because VMS combines character entropy h₂ ≈ 2.1 with positive length autocorrelation (AC = +0.160). Simple substitution preserves the autocorrelation structure of its source, and natural-language prose in the historically plausible source languages (Latin, Italian, German) shows negative or near-zero AC. Greshko (2025) demonstrated that verbose homophonic substitution replicates many VMS word-level statistics, but such systems fail on the length autocorrelation, the position-frequency gradient (−41.2 vs −1 to −2), and the gallows selection grammar. These gaps are systematic, not parametric.

Word-level encryption is excluded because word-to-word mutual information (0.45 bits, of which 98% derives from vocabulary frequencies) is an order of magnitude below the 3 to 5 bits expected of any word-boundary-preserving cipher applied to natural language. Independent word generation is excluded by the autocorrelation bound. The Cardan grille hypothesis (Rugg 2004; Schinner 2007) is excluded by morphological precision: grille-based generators produce 6.7 to 8.4% finite-state violations versus the manuscript's 0%.

Pure self-citation is excluded as a *complete* production mechanism. We implement Timm's algorithm and score it against 15 structural metrics (Supplement S5.6). Self-citation reproduces global distributional properties (Heaps law exponent, Zipf R-squared, word-length variance) but fails on metrics that depend on slot architecture: conditional character entropy overshoots by 32%, and the edit-distance-1 word network is nearly twice as dense as the VMS. The mechanism is real — Gaskell and Bowern (2022) experimentally confirmed it as the default strategy when humans produce meaningless text at scale — but it cannot reproduce the specific PGCS constraints. The words have parts, but not the right parts.

Character-level production models more generally — including positional bigram ledgers and character-walk generators — are excluded by over-generation (Table 2). A ledger of 584 positional character transitions licenses at least 5.3 million distinct token forms while failing to reach 48% of attested types, including high-frequency tokens. The attested PGCS quadruples compress the generative space 127-fold below the free product; character-level models expand it at least 700-fold above the attested vocabulary. Any production mechanism operating at the character level without slot-level constraints produces the wrong vocabulary by orders of magnitude.

### 4.4 The Generator Hierarchy

We construct a six-tier generator hierarchy to answer a specific question: what happens when self-citation operates within progressively more detailed structural constraints? Each tier implements the same core production mechanism (copy, modify, or create from a ledger of available words) but draws from ledgers of increasing specificity:

Tiers 1 and 2 operate without slot grammar. Tier 1 is character-level bigram generation. Tier 2 replaces character bigrams with calligraphic ductus groups. Tiers 3 through 6 incorporate PGCS structure at increasing resolution. Tier 3 introduces slot-level frequency matching. Tier 4 adds Currier A/B vocabulary differentiation. Tier 5 conditions on manuscript section with per-section copy and create rates. Tier 6 adds folio-restricted copy pools and suffix-to-prefix transition reweighting. Full specifications are in Supplement S3; the summary table below gives the key parameters.

| Tier | Name | Grammar | Copy / Modify / Create | Key constraint |
|------|------|---------|----------------------|----------------|
| T1 | Char Bigram | No | — | Character follower table (19 characters) |
| T2 | Ductus Groups | No | — | 5 calligraphic stroke groups |
| T3 | P70C Slot | Yes | 0.20 / 0.50 / 0.30 | 6,750 PGCS quintuples, global slot freq. |
| T4 | Currier A/B | Yes | 0.20 / 0.50 / 0.30 | Currier-language blending (70/30) |
| T5 | Section | Yes | varies | Per-section concentration (1.0–1.5) |
| T6 | Combined | Yes | 0.35 / 0.40 / 0.25 | Folio copy pool (350 tokens) + transitions |

We score each tier against 90 distributional metrics drawn from the Bowern-Gaskell benchmark suite and our own structural measures. The scoring is binary: a metric is satisfied if the generator's output falls within a fixed per-metric tolerance of the VMS value (tolerances documented in Supplement S3).

The results separate cleanly into two regimes, more cleanly than we expected. Tiers 1 and 2 (no grammar) reproduce 37 to 50 of 90 metrics. Tiers 3 through 6 (with grammar) reproduce 58 to 67. The jump from Tier 2 to Tier 3, when slot grammar is first introduced, is categorical — not the gradual improvement we anticipated from adding structural specification, but a discontinuity. No amount of additional specification within the grammar-free regime approaches the grammar-enabled floor, and no amount of additional specification within the grammar-enabled regime breaks substantially past the ceiling. Grammar is both necessary and sufficient to reach the grammar-constrained regime; it is not sufficient to reproduce the manuscript.

This result confirms Timm and Schinner's (2020) core insight. Self-citation is a powerful text production mechanism. When operating on raw vocabulary it accounts for approximately 50 of 90 distributional properties; when operating within the PGCS slot architecture it reaches 58 to 67. The grammar is what makes the difference, and the difference is not incremental.

The grammar itself is remarkably compact. The entire PGCS specification compresses to approximately 450 table entries: 25 functional glyphs, 65 prefix-gallows pairs, 66 suffix entries, 56 transition weights, and 231 core character bigrams (Supplement S1.6). This is, for illustrative comparison, roughly the capacity of a single VMS page. A scribe carrying a single leaf could, in principle, produce text satisfying 10 of 15 corpus-level metrics and the correct entropy scaling pattern across sections.

The generator ceiling (67/90 full-suite metrics, 88% of the Bowern-Gaskell 42-metric benchmark) approaches the VMS's own self-consistency across manuscript partitions (86%), suggesting that the remaining gap represents production decisions that no structural constraint captures. The BG benchmark percentage slightly exceeds self-consistency because the 42-metric subset excludes the metrics that drive the split-half disagreement; on the full 90-metric suite, the generator ceiling (74.4%) remains below self-consistency. The grammar jump is present in both BG-origin metrics (49–58% without grammar → 81–86% with) and custom structural metrics (34–53% → 47–66%), but grammar-enabled tiers essentially saturate the BG benchmark. The residual gap concentrates disproportionately in metrics not included in Gaskell and Bowern's original suite.

### 4.5 The Residual Gap

The generator hierarchy pinpoints where self-citation stops. Three metrics resist all six tiers simultaneously: word-length autocorrelation (AC(1) = +0.160), lexical repetition rate (0.008), and local vocabulary diversity (MATTR-25 = 0.919).

Gaskell and Bowern (2022) independently identified two of these three metrics among the ten most important features in their random forest classifier for distinguishing gibberish from meaningful text. Repeated words ranked sixth and word-length autocorrelation ranked tenth. This convergence between our generator hierarchy and their classifier was not designed: we selected metrics that jointly resist reproduction across generator tiers, and they selected metrics that best discriminate production mechanisms. The overlap strengthens the case that these metrics capture something structural about VMS production rather than artefacts of either method.

The mechanical basis of the residual gap is clear. Copying adjacent words raises AC but raises repetition and destroys diversity. Generating novel words preserves diversity but kills AC. The VMS achieves both simultaneously: its same-length-successive-word rate (4.47%) shows many consecutive word pairs sharing the same length without being the same word. Length correlation without repetition.

However, the VMS clusters with gibberish rather than with natural language on these metrics. Positive AC is a gibberish signature (Gaskell and Bowern 2022). Our generators, as they become more structurally sophisticated, move from the natural-language range toward the gibberish range. Closing the residual gap means becoming better gibberish, not capturing hidden information. The residual gap characterises the specific production mechanism — the VMS scribe(s) achieved a particular combination of length clustering and vocabulary novelty that our generators do not yet replicate — but it does not, by itself, distinguish meaningful text from a sufficiently sophisticated meaningless production process.

The bimodal vocabulary finding (§4.2) provides the additional constraint. The 97%/61% split (type-level hapax vs token-weighted non-hapax) is not a production-mechanism discriminator in Bowern's sense; it is a structural property internal to the manuscript. Whether the text is gibberish or meaningful, any production mechanism must explain why one-off words have systematically different morphological structure from repeated words. Self-citation does not predict this split. Neither do our generators produce it.

### 4.6 Section-Level Variation

Gaskell and Bowern (2022) explicitly acknowledged that their gibberish samples were "too short to test whether the higher-level structure of VMS pages and quires could also be produced by gibberish," and called this "a serious challenge to proponents of the hoax hypothesis." Our corpus-level analysis addresses this gap directly.

Nine manuscript sections share the same PGCS grammar but exhibit radically different distributional profiles:

| Section | N tokens | TTR | Hapax | AC(1) | Rep | ∅-core | ED% |
|---------|----------|-----|-------|-------|-----|--------|-----|
| Stars | 10,702 | 0.279 | 0.683 | +0.172 | 0.008 | 50.1% | 19.2% |
| Balneological | 6,859 | 0.219 | 0.664 | +0.124 | 0.012 | 63.5% | 27.9% |
| Herbal-B | 5,783 | 0.332 | 0.694 | +0.121 | 0.010 | 56.1% | 8.0% |
| Herbal-A | 4,033 | 0.355 | 0.708 | +0.076 | 0.009 | 56.9% | 0.0% |
| Pharmaceutical | 3,870 | 0.413 | 0.716 | +0.126 | 0.006 | 44.4% | 2.2% |
| Rosettes | 1,818 | 0.438 | 0.696 | +0.136 | 0.005 | 55.0% | 12.8% |
| Zodiac | 1,590 | 0.549 | 0.775 | +0.161 | 0.004 | 36.7% | 5.6% |
| Astronomical | 1,469 | 0.528 | 0.763 | +0.348 | 0.006 | 46.8% | 6.1% |
| Cosmological | 1,341 | 0.603 | 0.797 | +0.165 | 0.005 | 37.7% | 2.8% |

*Table 3. Per-section distributional profiles. All sections share the same PGCS grammar with zero cross-section violations. Distributional differences are discrete, not gradual.*

These differences are not gradual drift. The Balneological section's W1 gallows distribution is perfectly uniform (p = k = t, χ² = 0.00) while the Herbal section is p/f-dominant. Herbal-A contains zero ED tokens while the Balneological section reaches 27.9%. The qo-prefix is enriched 2.65× in the Balneological section relative to the Herbal section. The -edy suffix family dominates Balneological vocabulary while -aiin characterises Herbal. These are regime changes, not statistical scatter.

Timm and Schinner (2024) argued that vocabulary drift under self-citation could produce apparent section differentiation — if the scribe's available copying pool changes over time, frequency distributions will shift. This is true, but drift is gradual. The distributional profiles above show discrete breaks between sections, not smooth transitions. More importantly, 43.8% of Balneological vocabulary is unique to that section and absent from Herbal, Pharmaceutical, or Stars. Self-citation from a drifting pool does not explain why nearly half of a section's vocabulary appears nowhere else in the manuscript.

### 4.7 PGCS as a Shared Method

Fagin Davis (2020) identified five distinct scribal hands in the Voynich Manuscript through paleographic analysis. Timm and Schinner (2024) challenged this finding, arguing the variations could reflect a single scribe's evolution over time. Our distributional analysis supports Fagin Davis from a different angle: the distributional profiles across sections are too discrete to attribute to gradual drift. Sections attributed to different hands show radically different parameter settings — empty-core rates ranging from 37.7% to 63.5%, ED rates from 0% to 27.9% — while maintaining the same PGCS grammar with zero cross-hand violations.

If five scribes shared the same positional grammar, PGCS is not a personal habit or idiosyncrasy. It is a learnable, transmissible method — a shared protocol that multiple practitioners could follow with different distributional parameter settings. The production context that best fits this evidence is one in which the grammar was taught or codified, though the purpose — scribal training, encipherment, something else — cannot be determined from the statistics alone.

A single scribe generating intuitive gibberish — the strong form of Timm's model — would produce text with gradually shifting properties. Multiple scribes following a shared grammar would produce text with discrete distributional breaks between sections but consistent structural rules throughout. The VMS looks like the latter. The grammar's compactness (approximately 450 table entries) makes it something a scribe could learn from a colleague; the section-level variation looks less like drift than like different people working from the same recipe.

## 5. Discussion

### 5.1 Engaging Timm's Dynamic Hypothesis

Timm (2026) argues that VMS text is inherently dynamic, produced through iterative self-citation, and that static analytical frameworks fail because they cannot account for a text that was built by copying and modifying itself. Much of this framing aligns with our findings. The text is dynamic; section profiles and vocabulary drift across quires confirm it. Self-citation is real; our Tier 1 generator implements it and the 37 to 50 metrics it reproduces without grammar are consistent with Gaskell and Bowern's experimental finding that self-citation is the default strategy for producing meaningless text. Many static analyses do fail to account for the production process. Where we part company is on sufficiency.

The jump from 37 to 67 metrics when grammar is added is not incremental; it is categorical. Without grammar, no amount of tuning reaches the grammar-enabled floor. Timm (2026) acknowledges that the algorithm does not reproduce certain statistical properties. Our 90-metric suite quantifies exactly which those are.

We also identify a structural finding that self-citation does not predict: the bimodal vocabulary split. If self-citation were the complete production mechanism, hapax legomena should be accidental copy failures — words that happened not to be reused — and they should have the same internal structure as frequently copied words. The 97%/61% split shows they do not: hapax types overwhelmingly have filled cores (97%, type-level), while 61% of non-hapax running text consists of empty-core tokens. Something in the production process generates one-off words with different morphological structure from template words. This is not an impossibility proof. It is a constraint that any complete production model must satisfy.

### 5.2 Engaging Bowern and Gaskell

Gaskell and Bowern (2022) established the methodological framework against which VMS production hypotheses should be tested, and their experimental data remains the field's only direct evidence about how humans produce meaningless text. Our generator hierarchy is the systematic follow-up they proposed.

Their sole unreplicated metric — charbias_words_mean — corresponds to the PGCS grammar we formalise here. It is not typographic but morphological, with 210 falsifiable rules and zero held-out violations. Their gibberish samples were also too short to test section-level structure, and they flagged this as "a serious challenge to proponents of the hoax hypothesis." Our section analysis addresses this directly: nine sections with shared grammar and discrete distributional profiles. And their classifier identified VMS as more closely resembling gibberish than meaningful text — a finding our results are consistent with. The VMS's positive AC, low conditional entropy, and high repetition all fall on the gibberish side.

What follows from this is that the residual gap between our generators and VMS — concentrated in AC, repetition rate, and vocabulary diversity — does not prove hidden information. These are production-mechanism discriminators, and VMS falls on the gibberish side. We have not yet reverse-engineered the exact production recipe; that is not the same as showing the recipe encodes meaning. Whether it does is a question word-level metrics cannot answer.

### 5.3 Engaging Greshko

Greshko (2025) approached the VMS from the cipher side, constructing a historically plausible verbose homophonic substitution — the Naibbe cipher — that encrypts Latin and Italian into Voynich-like ciphertext. His cipher uses an expanded version of the Zattera slot grammar, which is structurally equivalent to our PGCS decomposition. His ciphertexts match VMS on many statistics simultaneously, but reproduce only approximately 30% of unique Voynich B word types and fail on the position-frequency gradient and gallows selection grammar. Our grammar-enabled generators reproduce 90–96% of attested VMS word types (union across 10 seeds of 37,465 tokens each), but over-generate by a factor of 2.5–3.1×, producing 19,000–24,000 unique types against the manuscript's 7,598. Without grammar, type recall drops to 22–24%. The grammar captures the space of possible words far more completely than the cipher; the cipher captures the selectivity more precisely.

Greshko's results and ours point to the same thing: the slot grammar is the structural backbone of VMS text, whether you call it a production grammar or a cipher structure. Neither his ciphertexts nor our generators fully replicate VMS, and the residual gaps in both cases say something about what the slot grammar determines and what it leaves open.

Where we differ is in what the grammar means. Greshko treats it as an encryption structure — the grammar determines how plaintext maps to ciphertext. We treat it as a production grammar that organises text generation regardless of whether the text carries semantic content. These readings are not mutually exclusive, and the statistical analyses neither of us performs can resolve the difference. What both approaches confirm independently is that the grammar is real, validated, and necessary.

### 5.4 The Constraint Space

The combined results define a constraint space that any valid production mechanism must occupy.

| Constraint | Value | What it excludes |
|-----------|-------|-----------------|
| h₂ | ≈ 2.1 bits | Random character sequences |
| Zipf R² | 0.915 | Flat-frequency generators |
| AC(1) | +0.160 (bound: 0.025) | IID word sampling |
| MI(quad; prev_sfx) | 0.757 bits | Sequential-independent generators |
| MI(quad; position) | 0.380 bits | Position-independent generators |
| PGCS coverage | 92.96% / 71.93% | Non-morphological systems |
| Held-out violations | 0/3,930 types | Overfitting / memorisation |
| Core-suffix coupling | Z = 215.0 | Independent slot filling |
| Information budget | 28.9% grammar / 71.1% content | Models conflating grammar with content |
| Hapax core fill rate | 97% filled types; 61% empty-core non-hapax tokens | Undifferentiated self-citation |
| Section-unique vocabulary | 43.8% (Balneological) | Gradual vocabulary drift |
| Generator ceiling | 67/90 (grammar) vs 50/90 (no grammar) | Grammar-free production |
| Residual gap | AC + repetition + diversity | All tested generators |
| Multiple hands | Shared grammar, discrete profiles | Single-scribe intuitive production |
| Over-generation | ≥701× (character), 0.68× (PGCS quads) | Character-level production models (Table 2) |

*Table 4. The constraint space. Any proposed production mechanism must simultaneously satisfy all rows.*

### 5.5 Limitations

The PGCS grammar outperforms alternatives by roughly 1,000× in decomposition error, but this is a relative comparison; it is the best available parse, not a proven unique one. PGCS is a statistical abstraction derived from co-occurrence patterns, not a claim about scribal cognition. Core findings have been validated against six transcription systems (Supplement S4); the four-slot architecture and information budget partition survive all six, with specific MI values shifting by an estimated ±0.05 bits. The structural conclusions are transcription-robust, though analyses in the main text use the ZLZI transcription throughout.

The generator hierarchy tests one mechanism family. Other production processes — verbose ciphers, constructed languages, compositional systems — could be tested against the same metric suite. Greshko (2025) has begun this work from the cipher side.

All significance tests operate on large N (37,465 tokens), so effect sizes matter more than p-values. Positional effects are small to medium (Cramér's V = 0.09 to 0.11 for suffix-section and prefix-section). Cross-slot couplings are medium to large (prefix-gallows V = 0.266; core-suffix V = 0.357, Cohen's w = 0.875; core-section V = 0.263). The word-length autocorrelation is 15.8× its 95% confidence interval under independence. The bimodal split at type level yields φ = 0.263 with odds ratio 7.2 (p = 2.3 × 10⁻¹¹⁶). We report mutual information alongside significance throughout to ensure effects are informationally substantive.

The bimodal vocabulary finding is descriptive. We report the 97%/61% split (type-level hapax vs token-weighted non-hapax) as an empirical constraint, not as evidence for or against meaning. Multiple explanations are compatible: semantic differentiation (content words vs function words, as in natural language), procedural differentiation (different generation strategies for template versus novel words), or some combination. The finding constrains the production mechanism without resolving the meaning question.

### 5.6 Falsification Criteria

Our claims are testable. A generator matching 68 or more of 90 metrics while simultaneously producing the correct hapax-core-fill pattern (97% filled-core hapax types, 61% empty-core non-hapax tokens) would falsify the ceiling claim. Discovery of productive PGCS-violating patterns would falsify the grammar itself. Demonstration that self-citation naturally produces the bimodal vocabulary split under conditions we have not tested would weaken the claim that the split requires a differentiated production process. All tests are empirically executable with the code and data we provide.

### 5.7 Future Work

The over-generation hierarchy (Table 2) and the 90-metric scoring framework provide a public benchmark against which competing production models can be evaluated. Any proposed mechanism — cipher, constructed language, compositional system — can be scored against the same metrics and its over-generation ratio compared to the attested PGCS quadruples. Greshko (2025) has begun this from the cipher side; comparable tests from the constructed-language and grille traditions would map the full mechanism space.

The residual gap (§4.5) isolates three metrics that resist all generator tiers: word-length autocorrelation, lexical repetition rate, and local vocabulary diversity. Closing this gap requires a production mechanism that achieves length clustering without repetition — a constraint that may point toward copy-pool management strategies not yet modelled. Whether the gap is closeable within a grammar-plus-self-citation framework, or requires an additional production component, remains an open empirical question.

The bimodal vocabulary split demands a mechanistic explanation. Among hapax types, 97% have filled cores; among non-hapax running text, 61% consists of empty-core tokens — a two-dimensional phenomenon in which rare types carry content (filled cores) while high-frequency types are structurally empty and dominate the text stream. Three candidate accounts are compatible with the data: semantic differentiation (content words versus function words, as in natural language), procedural differentiation (different generation strategies for template versus novel words), or a hybrid process. Targeted experiments — generating text under controlled conditions that vary the ratio of template reuse to novel production — could distinguish these accounts.

Core findings have been validated against six transcription systems (Supplement S4), confirming that the four-slot architecture and information budget survive cross-transcription testing with MI shifts of ±0.05 bits. A remaining open question is whether finer-grained findings — specific core inventories, suffix decomposition details, and the over-generation ratios in Table 2 — are equally stable, or whether some are sensitive to transcription-level glyph boundary decisions that the six-system test does not fully probe.

Preliminary distributional analysis using PPMI-weighted co-occurrence embeddings suggests that the four PGCS slots are not equal contributors to token distribution. The inner slot pair (gallows × core) constrains distributional behaviour at approximately 5× above chance baseline, while the outer pair (prefix × suffix) constrains at only 1.2×, with prefix showing near-zero independent constraining power. Gallows–core combinations show significant section specificity (χ² test, p < 0.01) for 66% of attested groups, covering 96% of all tokens, with each manuscript section exhibiting a distinctive gallows–core signature. These findings point toward a two-tier information architecture in which the inner slots encode content-domain information and the outer slots encode structural or positional function, to be reported in full elsewhere.

## 6. Conclusion

We formalised the PGCS slot grammar (92.96% coverage, zero held-out violations), decomposed VMS word selection into a grammar layer (28.9%) and content layer (71.1%), and constructed a six-tier generator hierarchy to test Timm and Schinner's self-citation hypothesis quantitatively, implementing the systematic generator programme proposed by Gaskell and Bowern (2022). An over-generation hierarchy quantifies the grammar's constraint power: character-level adjacency models license at least 700× more forms than attested, while attested PGCS quadruples compress the free combinatorial product 127-fold, with 75.4% of quadruples uniquely determining a single surface token.

Self-citation without grammar reproduces 37 to 50 of 90 distributional metrics. Adding the PGCS slot architecture raises this to 58 to 67 — a sharp discontinuity confirming that grammar is what separates partial from substantial replication. The residual gap concentrates in metrics independently identified by Gaskell and Bowern (2022) as production-mechanism discriminators, on which the VMS clusters with gibberish rather than natural language. This convergence validates these metrics as structurally informative but reframes the residual gap: it characterises VMS's specific production mechanism without resolving whether that mechanism encodes semantic content.

The PGCS grammar corresponds to the sole metric Gaskell and Bowern's experimental gibberish could not replicate, demonstrating that what they attributed to "typographic considerations" is a systematic morphological grammar. The bimodal vocabulary structure — hapax legomena with filled cores (97% of types), repeated types whose empty-core tokens dominate 61% of running text — provides an additional constraint that neither self-citation nor any tested generator reproduces, and that any complete production model must explain. Section-level distributional variation within shared grammar addresses the higher-level structural question Gaskell and Bowern identified as unresolved.

The VMS was produced using a shared morphological grammar (PGCS), combined with self-citation, by multiple scribes following a common method. This method generates two structurally distinct word classes — a high-frequency empty-core stratum dominating running text and a long tail of filled-core hapax types — and section-specific distributions within a unified framework. These properties go beyond what experimental gibberish, automated generators, or individual encipherment methods reproduce. As generators become more structurally sophisticated, they move toward the gibberish regime rather than away from it, consistent with Gaskell and Bowern's (2022) classification. Any cipher proposal must now explain not only the slot grammar but also why its ciphertext clusters with gibberish rather than with the plaintext language on production-mechanism metrics. PGCS characterises the production method. The meaning question remains open.

## Acknowledgments

The PGCS decomposition builds on foundational observations by Currier (1976), Tiltman (1967), and Stolfi (2005). Jorge Stolfi provided detailed feedback on entropy interpretation that substantially improved §3. The ZLZI transcription is maintained by René Zandbergen and colleagues. The Bowern-Gaskell benchmark suite (Bowern and Lindemann 2021; Gaskell and Bowern 2022) provided the statistical framework against which generators were evaluated. ChatGPT (GPT-4o) and Claude (Anthropic) were used as computational research assistants for statistical analysis, code development, and manuscript preparation; all analytical decisions and interpretations are the author's.

[^currier-bimodal]: Currier's A/B distinction is a between-section phenomenon; the bimodal vocabulary split (§4.2) is a within-section phenomenon. Every section contains both empty-core template words and filled-core hapax types; what varies is the ratio (37.7% to 63.5% empty-core rate). Currier's classification method detected this ratio difference plus the character-frequency shifts that follow from it. Variance decomposition confirms: Currier-level variance (0.001) is negligible once section is controlled (section variance = 0.029). Currier A/B is the statistical shadow of differential template-to-content ratios across sections, not evidence for two production systems.

## References

Bowern, C. and Lindemann, L. (2021). The linguistics of the Voynich Manuscript. *Annual Review of Linguistics*, 7, 285-308.

Cover, T. M. and Thomas, J. A. (2006). *Elements of Information Theory*. 2nd ed. Hoboken, NJ: Wiley.

Currier, P. (1976). Papers on the Voynich Manuscript. *New Research on the Voynich Manuscript: Proceedings of a Seminar*. Washington, D.C.

d'Imperio, M. E. (1978). *The Voynich Manuscript: An Elegant Enigma*. Fort Meade, MD: National Security Agency.

Fagin Davis, L. (2020). How many glyphs and how many scribes? Digital paleography and the Voynich Manuscript. *Manuscript Studies*, 5(1), 164-180.

Gaskell, D. E. and Bowern, C. (2022). Gibberish after all? Voynichese is statistically similar to human-produced samples of meaningless text. *CEUR Workshop Proceedings*, Vol-3313, International Conference on the Voynich Manuscript 2022. University of Malta.

Greshko, M. A. (2025). The Naibbe cipher: A substitution cipher that encrypts Latin and Italian as Voynich Manuscript-like ciphertext. *Cryptologia*. doi:10.1080/01611194.2025.2566408.

Landini, G. (2001). Evidence of linguistic structure in the Voynich Manuscript using spectral analysis. *Cryptologia*, 25(4), 275-295.

Matlach, V., Janečková, B. A., and Dostál, D. (2022). The Voynich Manuscript: Symbol roles revisited. *PLOS ONE*, 17(1), e0260948.

Montemurro, M. A. and Zanette, D. H. (2013). Keywords and co-occurrence patterns in the Voynich Manuscript: An information-theoretic analysis. *PLOS ONE*, 8(6), e66344.

Newbold, W. R. (1928). *The Cipher of Roger Bacon*. Philadelphia: University of Pennsylvania Press.

Reddy, S. and Knight, K. (2011). What we know about the Voynich Manuscript. In *Proceedings of the 5th ACL-HLT Workshop on Language Technology for Cultural Heritage, Social Sciences, and Humanities*, 78-86.

Rugg, G. (2004). An elegant hoax? A possible solution to the Voynich Manuscript. *Cryptologia*, 28(1), 31-46.

Schinner, A. (2007). The Voynich Manuscript: Evidence of the hoax hypothesis. *Cryptologia*, 31(2), 95-107.

Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379-423.

Stolfi, J. (2005). Voynich Manuscript word structure analysis. *Online manuscript*.

Strong, L. C. (1945). Anthony Askham, the author of the Voynich Manuscript. *Science*, 101(2633), 608-609.

Tiltman, J. (1967). The Voynich Manuscript: The most mysterious manuscript in the world. *NSA Technical Journal*, 12(3), 41-85.

Timm, T. and Schinner, A. (2020). A possible generating algorithm of the Voynich manuscript. *Cryptologia*, 44(1), 1-19.

Timm, T. and Schinner, A. (2024). The Voynich manuscript: Discussion of text creation hypotheses. *Cryptologia*, 48(4), 305-322.

Timm, T. (2026). The challenge of analyzing a dynamic text: Why the Voynich Manuscript resists conventional interpretation. *Cryptologia*.

Zandbergen, R. (2024). Voynich Manuscript transcription and analysis. *Online resource*. http://voynich.nu

Zipf, G. K. (1949). *Human Behavior and the Principle of Least Effort*. Cambridge, MA: Addison-Wesley.
