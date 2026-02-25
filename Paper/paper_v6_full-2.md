# Self-Citation Is Not Enough: Slot Grammar and the Impossibility Triple in the Voynich Manuscript

Edward Bozzard

*Submitted to Cryptologia, February 2026*

## Abstract

Timm and Schinner (2020) proposed that Voynich Manuscript text was produced by self-citation, the iterative copying and modification of previously written words. Timm (2026) argues that this process explains the text fully, rendering static analytical frameworks invalid. We test this claim quantitatively.

We formalise the four-slot PGCS grammar (Prefix + Gallows + Core + Suffix) that governs 92.96% of the manuscript's 37,465 tokens and decompose word selection into a grammar layer (28.9% of word-selection entropy) and a content layer (71.1%). We then construct a six-tier generator hierarchy implementing self-citation with increasingly specified ledgers (from raw vocabulary through full PGCS slot grammar with positional and sequential conditioning), scoring each tier against 90 distributional metrics extracted from the corpus.

Without slot grammar, generators reproduce 37 to 50 of 90 metrics. With grammar, they reach 58 to 67, a categorical jump confirming grammar as both necessary and sufficient to reach the grammar-constrained statistical regime. Three metrics, however, resist all generators simultaneously: word-length autocorrelation (AC(1) = +0.160), lexical repetition rate (0.008), and local vocabulary diversity (MATTR-25 = 0.919). This impossibility triple marks the quantitative boundary where self-citation stops. The text exhibits length-correlated sequential structure without the repetition that mechanical copying would produce.

Self-citation is a real component of VMS text production, but it operates within a grammar more constrained than self-citation alone generates.

**Keywords:** Voynich Manuscript, self-citation, slot grammar, generator hierarchy, impossibility triple, computational linguistics

## 1. Introduction

### 1.1 The Paradox

The Voynich Manuscript has resisted decipherment for over five centuries (d'Imperio 1978). Yet it is not random. Character-level bigram entropy (h₂ ≈ 2.1 bits) falls within the range of natural scripts. Word-frequency distributions follow Zipf's law (R² = 0.915; Zipf 1949). Vocabulary diversity matches natural language corpora, and five manuscript sections (herbal, pharmaceutical, biological, astronomical, and recipe) are distinguishable by vocabulary alone at 76 to 81% accuracy (Montemurro and Zanette 2013; Bowern and Lindemann 2021). By every surface metric, the text behaves like language.

The paradox is that beneath this surface, every attempt to recover lexical meaning has failed. No cipher proposal has produced readable text. No statistical clustering of tokens aligns with the manuscript's illustrations. Shuffling word order within folios does not degrade section classification, because the signal is purely vocabulary-based, not syntactic. The text has structure without recoverable semantics.

### 1.2 Previous Approaches

Three traditions dominate VMS scholarship. The cipher tradition (Newbold 1928; Strong 1945; numerous modern proposals) assumes the text encodes natural language through substitution or transposition; all published decipherments have failed reproducibility tests (d'Imperio 1978; Reddy and Knight 2011). The linguistic tradition (Currier 1976; Tiltman 1967; Landini 2001; Bowern and Lindemann 2021) documents statistical regularities, including positional preferences, character-level constraints, and word-frequency distributions, without claiming decipherment. More recently, Matlach, Janečková, and Dostál (2022) identified symbol-role patterns using information-theoretic analysis, and Greshko (2025) demonstrated that a verbose homophonic substitution cipher can replicate many VMS word-level statistics simultaneously while remaining fully decipherable and historically plausible. The hoax tradition (Rugg 2004; Schinner 2007) proposes that the text was generated mechanically to simulate language without carrying meaning.

Each tradition captures part of the picture. The text is too structured for a simple hoax yet too anomalous for any proposed cipher; it matches natural language statistics yet resists semantic analysis. Timm and Schinner (2020) proposed a concrete mechanism that cuts across these boundaries: self-citation, the iterative copying and modification of previously written words. Their algorithm unifies the hoax and stochastic traditions by showing how a scribe could produce VMS-like text without encoding meaning. Timm and Schinner (2024) extended this argument at the Malta conference, and Timm (2026) makes the strongest claim yet: self-citation explains the text fully, and all static analytical frameworks are invalid because the text is inherently dynamic. This paper tests that claim.

### 1.3 This Paper

We approach the question with a single method applied to a single corpus. Section 2 formalises the PGCS slot grammar that any production mechanism, including self-citation, must operate within. Section 3 quantifies the grammar/content boundary through an information budget that partitions word-selection entropy into a grammatical component (28.9%) and a lexical content component (71.1%). Section 4 builds a six-tier generator hierarchy that extends Timm's algorithm with increasingly specified structural ledgers, scoring each against 90 distributional metrics. Section 5 discusses the results in direct conversation with Timm (2026) and defines the constraint space bounding admissible production mechanisms. Full generator specifications, metric definitions, and scoring code are available at [GitHub/Zenodo DOI].

## 2. The PGCS Decomposition

### 2.1 Corpus

All analyses use the Zandbergen-Landini-Zandbergen-IVTFF (ZLZI) transcription of the Voynich Manuscript (Zandbergen 2024), comprising 37,465 tokens and 7,598 types across 226 folios. This transcription system uses Extended Voynich Alphabet (EVA) characters. Sensitivity analysis under 5% character corruption (Supplement S4) confirms that structural findings are robust to transcription-level noise, though all analyses are conditioned on a single transcription system.

### 2.2 The Four-Slot Model

The Prefix-Gallows-Core-Suffix (PGCS) model decomposes each token into four ordered slots.

The Prefix slot (8 types) occupies word-initial position before any gallows character. The critical innovation of PGCS is the reassignment of *ch* and *sh* from gallows to prefix status. These characters behave distributionally as prefixes: they appear at word-initial position, combine freely with following gallows characters, and account for 66.6% of all tokens containing them (24,964 tokens). Under previous three-slot models, *ch* and *sh* were grouped with gallows characters, obscuring the true slot structure.

The Gallows slot (9 types) contains the tall characters *k*, *t*, *p*, *f* and their bench variants *ckh*, *cth*, *cph*, *cfh*, plus the rare *m*. The Core slot (open class) carries the central morphemic material; critically, 52.7% of tokens have empty cores. The Suffix slot (33 types) is a closed inventory of terminal material with highly regular distributional properties.

The reclassification of *ch* and *sh* from gallows to prefix is the most consequential boundary decision, affecting 24,964 tokens. It produces the 52.7% empty-core rate, nearly halves core-suffix mutual information (from 1.860 to 0.976 bits), and recasts common words like "chedy" as function words [ch | ∅ | ∅ | edy] rather than content words [∅ | ch | e | dy]. The four-slot design is further motivated by the scale difference between gallows (9 types) and core (2,001 types), by the prefix-gallows association being the strongest pairwise coupling in the system (Cramér's V = 0.266), and by the fact that the empty-core rate is invisible under any model that folds gallows into core.

### 2.3 Validation

We formalise the PGCS grammar as 210 rules: 81 character-sequence constraints, 52 pair-adjacency rules, 41 suffix rules, and 36 prefix rules. The grammar derivation methodology, including rule discovery from character co-occurrence patterns, the data-driven reclassification of *ch* and *sh*, and the train/test protocol with 40% held-out data frozen at the P69 stage, is documented in Supplement S1. The merged character grammar achieves character coverage of 92.96% and fully parses 71.93% of word types (99.87% partial coverage). The uncovered 7.04% of characters occur in rare bigram contexts concentrated in low-frequency types; they do not form productive patterns of their own. Every attested token receives a valid four-slot parse.

To test generalisation, we held out 40% of the corpus by folio. The held-out set contains 3,930 unique types not seen during rule development; zero produce PGCS violations. The grammar generalises perfectly to unseen data. Position-conditioned quintuples (prefix, gallows, core-class, suffix-family, line-position) provide a further test: 6,750 observed quintuples from 37,465 tokens represent a 292-fold compression from the theoretical combinatorial space, again with zero validation failures.

Comparison against all alternative slot assignments yields a decomposition error of 0.001 bits for PGCS versus 1.074 bits for the next-best alternative, a roughly 1,000-fold difference. The most competitive alternatives either fold *ch*/*sh* back into gallows (producing a 3-slot model that masks the empty-core rate and inflates core-suffix MI to 1.860 bits) or merge gallows into core (producing a 3-slot model where the core slot spans 9 to 2,001 types, collapsing the cardinality distinction that separates closed-class from open-class behavior). Both alternatives degrade held-out generalisation and produce higher redundancy between slots. Adversarial tests targeting circularity, overfitting, and transcription dependence (Supplement S4) likewise fail to produce degradation, supporting the interpretation that the slot architecture reflects underlying structure rather than model-specific fitting.

## 3. The Grammar-Content Boundary

This section builds the paper's central quantitative result: a partition of word-selection entropy into a grammatical component and a content component. We establish this in three stages. Slot-level analysis reveals how much structure the four-slot decomposition captures within individual words. Sequential and positional analysis shows that structural constraints extend across word boundaries and through line position. These axes are then combined into a single information budget.

### 3.1 Within-Word Structure

The chain rule of entropy (Shannon 1948; Cover and Thomas 2006) guarantees H(word) = H(P) + H(G|P) + H(C|P,G) + H(S|P,G,C) for any lossless decomposition. The empirical finding lies in the gap between this sum and the sum of marginal slot entropies. Unconditional slot entropies sum to 13.171 bits, exceeding H(word) = 10.311 by 2.860 bits (21.7% redundancy). PGCS slots are approximately 78% independent, with the remaining 22% carried primarily by the core-suffix association (MI = 0.976 bits, full 2,001-type core), followed by prefix-core (MI = 0.428 bits) and prefix-gallows (MI = 0.393 bits).

More than half of all tokens (52.7%) have empty cores, composed entirely from closed-class inventory items. This rate varies by section, from 37.7% (Cosmological) to 63.5% (Balneological). Unlike natural language, where content words dominate running text, this manuscript's vocabulary is majority-functional. Second-order character entropy (h₂ = 2.13 bits) nonetheless falls within the natural language range documented by Bowern and Lindemann (2021), ranging from 1.96 (biological) to 2.23 (pharmaceutical) across sections.

### 3.2 Across-Word Structure

If words were selected independently, even from position-specific pools, adjacent word lengths would show negligible correlation. In fact they are positively correlated: AC(1) = +0.160. We prove (Supplement S5.4) that under independent sampling the expected autocorrelation magnitude is bounded by approximately 0.025 given VMS parameters. The observed value exceeds this bound by more than sixfold, and simulation with 10,000 replications confirms that no replication exceeds |AC| = 0.028. Independent word generation (where each word is drawn independently, even from position-specific pools) is formally excluded as a production mechanism.

Having established that words are not independent, we can ask how they depend on each other. The answer is a suffix-to-prefix transition grammar. The conditional distribution P(prefix_{N+1} | sfx_fam_N) across all consecutive word pairs yields an 8×8 transition matrix with strong directional biases: Y-suffix → qo-prefix (26.0%, 1.9× enrichment), BARE-suffix → ∅-prefix (39.5%, 1.8×), N-suffix → o-prefix (29.7%), and R-suffix → o-prefix (26.1%). Line-initial tokens show a distinct distribution with elevated y-prefix (13.4%), s-prefix (9.3%), and d-prefix (14.3%).

This transition grammar contributes MI(quad; prev_sfx) = 0.757 bits beyond section and position conditioning combined, where "quad" denotes the four-slot combination (prefix, gallows, core-class, suffix-family). This makes it the single largest additional predictive axis. It also identifies the structural pathway behind the length autocorrelation: suffix-family constrains the following prefix, and prefix constrains word length, so the coupling propagates through the PGCS slot system rather than through direct length dependence. The effect resets at line boundaries, consistent with within-line AC (0.151) exceeding cross-line AC (0.062).

The positional grammar confirms Currier's (1976) observation that the line is a functional unit, not merely a scribal convenience. Word selection correlates with line position at MI(quad; position) = 0.380 bits, and line structure follows a three-zone pattern. The opener zone is enriched for d-prefix (15.3% vs 10.1%, Z = 12.5) and depleted for empty cores (39% vs 57%). The closer zone carries the M-suffix line-ending marker at 14.9% versus 1.8% at the penultimate position (Z = 57.8), by far the strongest positional signal in the manuscript. The middle zone is the default register. First-line-of-paragraph openers (226 tokens) are particularly striking: 71.7% carry ∅-prefix, 84.5% bear gallows, and only 12.4% have empty cores. These proportions are strongest in the herbal and stars sections and weaker in the zodiac and cosmological folios, suggesting that the paragraph-marker convention is section-dependent. This paragraph-marker class has no known analogue in published analyses of the Voynich Manuscript, and is consistent with a notation system that marks discourse boundaries through slot configuration rather than lexical choice. Position conditioning eliminates 58.6% of spurious (token, position) assignments, constraining where each type can appear without changing which types the grammar accepts.

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

The largest single increment comes from the sequential transition axis (0.757 bits), which identifies suffix-to-prefix coupling as the dominant sequential structure, consistent with the autocorrelation bound established in §3.2. The paragraph flag contributes negligibly (0.030 bits) despite its striking distributional signature, because the 226 paragraph-initial tokens represent only 0.6% of the corpus. Quire identity adds 0.518 bits beyond section, reflecting within-section vocabulary variation across production units, consistent with cross-quire lexical clustering documented independently through paleographic analysis (Fagin Davis 2020).

The 71.1% unexplained entropy is not noise. It is the content signal: the specific lexical choices that distinguish one passage from another within the same section, position, and grammatical context. "Content" here denotes unexplained statistical variability in word selection, not recoverable semantic meaning; the information budget is agnostic about whether this residual encodes language, carries structured nonsense, or reflects some intermediate state. The grammar (29%) is fully reproducible without semantic encoding. The content (71%) is what resists recovery. The remainder of this paper tests how far self-citation can reach into that content layer.

## 4. Self-Citation and Beyond

### 4.1 Structural Complexity of the PGCS Lexicon

The grammar produces structured text. Before we can test how far self-citation reaches, we need to establish what any production mechanism must reproduce.

The strongest coupling in the dataset is between classified core-class and suffix-family selection (MI = 0.470 bits; Z = 173.8 against the null of independence): certain cores preferentially take certain suffixes, so VMS words are not randomly assembled. This coupling interacts with position. The 142 cores occurring ten or more times partition into opener-biased, closer-biased, and position-neutral classes (χ² = 2608.6, p ≈ 0), and even visually similar cores (edit distance = 1) produce significantly different suffix profiles in 73.6% of 280 tested pairs. The grammar tracks morphemic identity at fine granularity.

At the vocabulary level, the lexicon stratifies into functional and content-bearing components. A set of whole-word tokens behave as function-word candidates, with coefficient of variation below 0.37 across sections and typically complex PGCS structure with no core (a full list is available in the repository). At the other extreme, a comparable number of cores concentrate in one or two sections (CV > 0.8), recapitulating the vocabulary differentiation that drives section classification but localising it to the core slot. The Currier A/B "languages" similarly manifest as differential word selection within a shared grammar rather than as distinct grammars.

The suffix decomposition reveals a further structural property. The coupling between vowel prefix and terminal consonant within suffixes is strong (I(VP; terminal) = 1.478 bits; NMI = 0.644), meaning that not all vowel-prefix/terminal combinations are equally available. The suffix operates as a constrained combinatorial system rather than a free product of independent axes. A stochastic generator implementing the PGCS slot architecture reproduces this coupling (normalised mutual information NMI = 0.615, where NMI = MI/min(H(X), H(Y)) ranges from 0 to 1), while generators lacking slot structure produce substantially weaker coupling (NMI = 0.403 to 0.473). This bounds the minimum structural complexity required to produce VMS-like text: mechanisms must enforce within-suffix co-occurrence constraints, not merely character-level bigram legality.

### 4.2 Mechanism Exclusions

The information budget and structural complexity documented above formally exclude several classes of production mechanism (full derivations in Supplement S5). Each exclusion eliminates a region of mechanism space by identifying a specific empirical constraint that the candidate process cannot satisfy. Taken together, they define a multidimensional admissible region: any viable production hypothesis must simultaneously respect the entropy budget, the autocorrelation sign, the slot-coupling strength, and the positional precision documented above.

Monoalphabetic substitution over natural-language prose is excluded because VMS combines character entropy h₂ ≈ 2.1 with positive length autocorrelation (AC = +0.160). Simple substitution preserves the autocorrelation structure of its source, and natural-language prose in the historically plausible source languages (Latin, Italian, German) shows negative or near-zero AC. No monoalphabetic cipher over tested source corpora can produce the VMS's positive value; the exclusion holds for any source with non-positive length autocorrelation. Greshko (2025) demonstrated that verbose homophonic substitution replicates many VMS word-level statistics, but such systems fail on the length autocorrelation, the position-frequency gradient (−41.2 vs −1 to −2), and the gallows selection grammar. These gaps are systematic, not parametric.

Word-level encryption is excluded because word-to-word mutual information (0.45 bits, of which 98% derives from vocabulary frequencies) is an order of magnitude below the 3 to 5 bits expected of any word-boundary-preserving cipher applied to natural language. Independent word generation (IID sampling, where each word is drawn independently from a fixed or position-specific pool) is excluded by the autocorrelation bound: the suffix-to-prefix transition grammar identifies the specific coupling mechanism, and no IID model can replicate it. The Cardan grille hypothesis (Rugg 2004; Schinner 2007) is excluded by morphological precision: we implemented grille-based generators with parameters matching Rugg's published specification, and they produce 6.7 to 8.4% finite-state violations versus the manuscript's 0%.

Pure self-citation, the iterative copying and modification of previously written words (Timm and Schinner 2020; Timm 2026), is excluded as a *complete* production mechanism. We implement Timm's algorithm and score it against 15 structural metrics (Supplement S5.6). Self-citation reproduces global distributional properties (Heaps law exponent, Zipf R-squared, word-length variance) but fails on metrics that depend on slot architecture: conditional character entropy overshoots by 32%, and the edit-distance-1 word network is nearly twice as dense as the VMS. The mechanism is real. Gaskell and Bowern (2022) experimentally confirmed self-citation as the default strategy when humans produce meaningless text at scale. Self-citation does produce words with internal structure (any copying process preserves some beginning/middle/end patterning), but it cannot reproduce the specific PGCS constraints: suffix-bearing tokens drop from 93.5% to 59.9%, within-suffix coupling falls from NMI 0.644 to 0.473, and positional precision degrades (q-initial from 99.4% to 89.8%). The words have parts, but not the right parts.

Having excluded competing mechanisms, we test how far self-citation can reach when supplied with the correct grammar.

### 4.3 The Generator Hierarchy

We construct a six-tier generator hierarchy to answer a specific question: what happens when self-citation operates within progressively more detailed structural constraints? Each tier implements the same core production mechanism (copy, modify, or create from a ledger of available words) but draws from ledgers of increasing specificity:

Tiers 1 and 2 operate without slot grammar. Tier 1 is character-level bigram generation: each token is produced character by character following a bigram chain trained on the first folio, capturing only character co-occurrence patterns. Tier 2 replaces character bigrams with calligraphic ductus groups (bench, gallows, loop, ligature), generating tokens by following group-to-group transitions and selecting character exemplars within each group — capturing pen-stroke habits without morphological awareness. Tiers 3 through 6 incorporate PGCS structure at increasing resolution. Tier 3 introduces slot-level frequency matching. Tier 4 adds Currier A/B vocabulary differentiation. Tier 5 conditions on manuscript section with per-section copy and create rates. Tier 6 adds folio-restricted copy pools and suffix-to-prefix transition reweighting, the full model described in Supplement S3.

We score each tier against 90 distributional metrics drawn from the Bowern-Gaskell benchmark suite and our own structural measures, covering character-level statistics, word-level distributions, sequential dependencies, and positional patterns. The scoring is binary: a metric is satisfied if the generator's output falls within a fixed per-metric tolerance of the VMS value (tolerances documented in Supplement S3 and the scoring code).

The results separate cleanly into two regimes. Tiers 1 and 2 (no grammar) reproduce 37 to 50 of 90 metrics. Tiers 3 through 6 (with grammar) reproduce 58 to 67. The jump from Tier 2 to Tier 3, when slot grammar is first introduced, is categorical. No amount of additional specification within the grammar-free regime approaches the grammar-enabled floor, and no amount of additional specification within the grammar-enabled regime breaks substantially past the ceiling. Grammar is both necessary and sufficient to reach the grammar-constrained regime; it is not sufficient to reproduce the manuscript.

This result confirms Timm and Schinner's (2020) core insight. Self-citation is a powerful text production mechanism. When operating on raw vocabulary it accounts for approximately 50 of 90 distributional properties; when operating within the PGCS slot architecture it reaches 58 to 67. The grammar is what makes the difference, and the difference is not incremental. It is a sharp discontinuity in statistical fidelity: no intermediate configuration (partial grammar, partial slot structure) produces an intermediate score.

The grammar itself is remarkably compact. The entire PGCS specification compresses to approximately 450 table entries: 25 functional glyphs, 65 prefix-gallows pairs, 66 suffix entries (decomposed into vowel-prefix and terminal components), 56 transition weights, and 231 core character bigrams (Supplement S1.6). This is, for illustrative comparison, roughly the capacity of a single VMS page — though this analogy is meant to convey compactness, not to imply any particular scribal practice. A scribe carrying a single leaf could, in principle, produce text satisfying 10 of 15 corpus-level metrics and the correct entropy scaling pattern across sections. What that scribe could not produce, even with this grammar, is the impossibility triple described below.

The generator ceiling (67/90 full-suite metrics, 88% of the Bowern-Gaskell 42-metric benchmark) approaches the VMS's own self-consistency across manuscript partitions (86%), suggesting that the remaining gap represents genuine lexical decisions that no structural constraint captures. The BG benchmark percentage slightly exceeds self-consistency because the 42-metric subset excludes the impossibility metrics and hapax-sensitive measures that drive the split-half disagreement; on the full 90-metric suite, the generator ceiling (74%) remains well below self-consistency.

All generator code and scoring scripts are available at [GitHub/Zenodo DOI], and every number reported here can be reproduced in under 60 seconds.

### 4.4 The Impossibility Triple

The generator hierarchy reveals not only how far self-citation reaches but precisely where it stops. Three metrics resist all six tiers simultaneously, and the reason they resist is not a matter of parameter tuning but a structural incompatibility among three measurable constraints.

Word-length autocorrelation (AC(1) = +0.160) requires that adjacent words have correlated lengths. In natural language this correlation is typically negative (long words followed by short ones); in the VMS it is positive and strong, exceeding the independence bound by more than sixfold. The most obvious way to produce positive AC through self-citation is to copy nearby words, but copying raises the second metric: lexical repetition rate (VMS = 0.008, meaning that only 0.8% of adjacent word pairs are identical). This is extraordinarily low. And any generator that produces novel (non-copied) words to keep repetition down sacrifices the third metric: local vocabulary diversity (MATTR-25 = 0.919, compared with 0.919 to 0.967 across our generators), which requires that local 25-word windows maintain high type diversity.

The mechanical conflict is clear. Copying adjacent words raises AC but destroys diversity and raises repetition. Generating novel words preserves diversity but kills AC. The VMS achieves both simultaneously, and our same-length-successive-word (SLSW) rate of 4.47% shows how: the text contains many consecutive word pairs that share the same length but are not the same word. Length correlation without repetition. This is the signature that no copy-and-modify process reproduces.

The impossibility triple is not about the generator hierarchy failing to find the right parameters. We do not claim a mathematical impossibility proof. We claim an empirical ceiling that persists across all tested architectures, and whose structural basis, the copying/novelty tradeoff, suggests it is not an accident of our particular implementations. On one side of this boundary sits mechanical text production (copying, modifying, recombining). On the other, something that makes word-by-word lexical decisions correlated in length but independent in identity. Whatever that something is, it lies beyond self-citation.

This claim is falsifiable. A generator matching 68 or more of 90 metrics, including all three impossibility metrics simultaneously, would refute the ceiling. We provide the scoring code and metric definitions to make this test executable.

## 5. Discussion

### 5.1 Engaging Timm's Dynamic Hypothesis

Timm (2026) argues that VMS text is inherently dynamic, produced through iterative self-citation, and that static analytical frameworks fail because they cannot account for a text that was built by copying and modifying itself. We agree on three of four points and disagree on the most consequential one.

We agree that the text is dynamic. Section profiles, vocabulary drift across quires, and the 0.518-bit contribution of production-unit identity to the information budget all point to a text that changed as it was written. We agree that self-citation is real. Our Tier 1 generator implements it directly, and the 37 to 50 metrics it reproduces without any grammar are consistent with Gaskell and Bowern's (2022) experimental finding that self-citation is the default strategy for generating meaningless text at scale. We agree that many static analyses fail to account for the production process. The Currier A/B distinction, for instance, is better understood as differential word selection within a shared grammar than as evidence for two separate scribal traditions (though it may reflect both).

We disagree that self-citation is sufficient. The jump from 37 to 67 metrics when grammar is added is not incremental; it is categorical. Timm (2026) acknowledges that the algorithm does not reproduce certain statistical properties of the text. Our 90-metric suite quantifies exactly which observations those are and why they resist: the impossibility triple is not about subtlety but about a mechanical conflict at the heart of copy-and-modify production. No amount of parameter tuning resolves a conflict between copying (which raises both AC and repetition) and novelty (which preserves diversity but kills AC). The VMS resolves it, and self-citation does not.

### 5.2 The Constraint Space

The combined results define a narrow constraint space that any valid production mechanism must occupy.

Calibration against a historically grounded scribal model, implementing calligraphic ductus constraints and experimentally validated self-citation rates with no VMS-derived parameters, establishes a baseline for what 15th-century scribal practice alone can produce. This baseline matches VMS character-level statistics (mean word length within 0.3%, conditional entropy within 13%, edit-distance network degree within 1.5%) but falls short on within-suffix coupling (NMI = 0.403 vs 0.644, a 37% deficit) and lexical diversity (hapax rate, Zipf exponent, MATTR). The constraint space therefore separates into a structural layer reproducible by scribal training and a combinatorial layer requiring additional production constraints equivalent to the PGCS slot architecture.

| Constraint | Value | What it excludes |
|-----------|-------|-----------------|
| h₂ | ≈ 2.1 bits | Random character sequences |
| Zipf R² | 0.915 | Flat-frequency generators |
| AC(1) | +0.160 (bound: 0.025) | IID word sampling |
| MI (word-word) | 0.45 bits (98% frequency) | Word-level encryption |
| MI(quad; prev_sfx) | 0.757 bits | Sequential-independent generators |
| MI(quad; position) | 0.380 bits | Position-independent generators |
| Placement precision | 58.6% spurious pairs eliminated | Unconstrained positional assignment |
| Position-frequency gradient | −41.2 | Stationary generators |
| PGCS coverage | 92.96% / 71.93% | Non-morphological systems |
| Held-out violations | 0/3,930 types | Overfitting / memorisation |
| Core-suffix coupling | Z = 173.8 | Independent slot filling |
| Core substitutability | 73.6% distinguishable | Free equivalence classes |
| Function words (CV<0.37) | Present | Purely content-bearing text |
| Section-variable cores (CV>0.8) | Present | Homogeneous notation |
| Information budget | 28.9% grammar / 71.1% content | Models conflating grammar with content |
| Impossibility triple | AC + repetition + diversity | All tested generators |

*Table 2. The constraint space. Any proposed production mechanism must simultaneously satisfy all rows.*

### 5.3 Limitations

The PGCS grammar outperforms alternatives by roughly 1,000× in decomposition error, but this is a relative comparison; it is the best available parse, not a proven unique one. PGCS is a statistical abstraction derived from co-occurrence patterns, not a claim about scribal cognition; whether the scribe(s) conceptualised production in terms of four slots, or followed a different cognitive procedure that happens to produce four-slot-compatible output, is outside the scope of this analysis. All analyses use a single transcription system (ZLZI), and while sensitivity analysis under character corruption confirms robustness of structural findings (Supplement S4), the dependence on a single transcription should be noted. The four-slot architecture and information budget partition are expected to be transcription-robust because they depend on distributional clustering rather than exact character identity; specific MI values and transition weights would shift by an estimated ±0.05 bits under alternative transcriptions, but the 28.9%/71.1% partition and the impossibility triple should persist. Testing against alternative transcription systems (e.g., v101, Takahashi) would strengthen the claims but has not yet been performed.

The generator hierarchy tests one mechanism family. Other production processes (verbose ciphers, constructed languages, compositional systems) could be tested against the same metric suite, and we would welcome such tests. Semantic analyses (section classification, vocabulary clustering, illustration-based groupings) were conducted separately and uniformly fail to detect recoverable content, consistent with the impossibility triple; full methodology is available on request.

All significance tests operate on large N (37,465 tokens), so effect sizes matter more than p-values. They are small to medium (Cramér's V = 0.1 to 0.34 for positional effects). We report mutual information alongside significance throughout to ensure effects are informationally substantive, not merely significant.

### 5.4 Falsification Criteria

Our claims are testable. As stated in §4.4, a generator matching 68 or more of 90 metrics while simultaneously satisfying all three impossibility metrics would falsify the ceiling claim. Discovery of productive PGCS-violating patterns (types that systematically violate the slot architecture while participating in the same distributional regularities as compliant types) would falsify the grammar itself. Both tests are empirically executable with the code and data we provide.

## 6. Conclusion

We formalised the PGCS slot grammar (92.96% coverage, zero held-out violations), decomposed VMS word selection into a grammar layer (28.9%) and content layer (71.1%), and constructed a six-tier generator hierarchy to test Timm and Schinner's self-citation hypothesis quantitatively.

Self-citation without grammar reproduces 37 to 50 of 90 distributional metrics. Adding the PGCS slot architecture raises this to 58 to 67, a categorical jump confirming that grammar is both necessary and sufficient to reach the grammar-constrained statistical regime, though not to reproduce the manuscript itself. Three metrics resist all generators simultaneously: word-length autocorrelation, repetition rate, and local lexical diversity. This impossibility triple marks the quantitative boundary between the mechanical production processes we have tested and whatever process produced the remaining signal. The text exhibits length-correlated sequential structure without the repetition that copying would produce, and no generator we tested, or that we can see how to construct, resolves this conflict.

Timm (2026) argues that VMS text is fully explained by self-citation. Our results confirm self-citation as a real and powerful component of VMS production but show it operates within a grammar more constrained than self-citation alone generates, and against a content signal that no copy-and-modify process reproduces. The constraint space is specific, quantitative, and falsifiable. Any proposed production mechanism, whether it encodes meaning or generates nonsense, must crack the impossibility triple to claim it explains the text.

## Acknowledgments

The PGCS decomposition builds on foundational observations by Currier (1976), Tiltman (1967), and Stolfi (2005). Jorge Stolfi provided detailed feedback on entropy interpretation that substantially improved §3. The ZLZI transcription is maintained by René Zandbergen and colleagues. The Bowern-Gaskell benchmark suite (Bowern and Lindemann 2021; Gaskell and Bowern 2022) provided the statistical framework against which generators were evaluated. ChatGPT (GPT-4o) and Claude (Anthropic) were used as computational research assistants for statistical analysis, code development, and manuscript preparation; all analytical decisions and interpretations are the author's.

## References

Bowern, C. and Lindemann, L. (2021). The linguistics of the Voynich Manuscript. *Annual Review of Linguistics*, 7, 285-308.

Cover, T. M. and Thomas, J. A. (2006). *Elements of Information Theory*. 2nd ed. Hoboken, NJ: Wiley.

Currier, P. (1976). Papers on the Voynich Manuscript. *New Research on the Voynich Manuscript: Proceedings of a Seminar*. Washington, D.C.

d'Imperio, M. E. (1978). *The Voynich Manuscript: An Elegant Enigma*. Fort Meade, MD: National Security Agency.

Fagin Davis, L. (2020). How many scribes? A paleographic assessment of the Voynich Manuscript. *Yale University Library Gazette*, Occasional Paper.

Gaskell, T. and Bowern, C. (2022). Gibberish after all? Voynich Manuscript scribes show patterns consistent with meaningful text generation. *Proceedings of the Linguistic Society of America*, 7(1), 5279.

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

Timm, T. and Schinner, A. (2020). A possible linguistic classification of the Voynich Manuscript. *Cryptologia*, 44(3), 216-230.

Timm, T. and Schinner, A. (2024). The Voynich manuscript: Discussion of text creation hypotheses. *Cryptologia*, 48(4), 305-322.

Timm, T. (2026). The challenge of analyzing a dynamic text: Why the Voynich Manuscript resists conventional interpretation. *Cryptologia*.

Zandbergen, R. (2024). Voynich Manuscript transcription and analysis. *Online resource*. http://voynich.nu

Zipf, G. K. (1949). *Human Behavior and the Principle of Least Effort*. Cambridge, MA: Addison-Wesley.
