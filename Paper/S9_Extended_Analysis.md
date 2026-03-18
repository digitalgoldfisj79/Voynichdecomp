# Supplement S9: Extended Analysis

Detailed analyses moved from the main text for brevity.

## S9.1 f57v Ring-by-Ring Analysis

Ring 1 (innermost) contains 29 individual components, evenly spaced and
visually unlike any word-bearing line in the manuscript, followed by 4
complete words and 3 final characters. Among the 29 components: 26
single characters and three multi-character units. The units are *aiin*
(the manuscript\'s most frequent word-ending, appearing in 10.2% of all
tokens) and *ar* (the fourth most frequent, 7.1%). They are listed at
the same spacing as single characters because in this system they are
atomic: pre-assembled blocks. The final tokens (*dar*, *teodar*,
*otodal*, *sheky*, *oteeody*) demonstrate how the blocks combine into
complete words.

Ring 2 opens with 4 words, closes with 4 words and a terminal mark.
Between them, a deliberate alternating pattern: clusters of 1 to 4 words
separated by isolated single characters. Nine such characters appear
across all three full-coverage transcriptions; in the JGLI transcription, all nine are
drawn from Ring 3\'s inventory (9/9 match). In ZLZI and TTLI, eight of
nine match, with the ninth read as \'s\'. Words alongside the characters
they are built from.

Ring 3 contains purely evenly-spaced individual glyphs, no word
groupings. Four repetitions of a 12-character sequence. At position 9,
*f* appears in periods 1 and 2 and *p* in periods 3 and 4. This is the
character inventory, and the variation demonstrates that *f* and *p* are
interchangeable: they occupy the same structural position in the gallows
slot.

Ring 4 (outermost) is almost entirely words. Normal manuscript running
text. The finished product.

The PGCS grammar was derived entirely from the running text of the
manuscript's 226 text-bearing pages; f57v is a diagram page excluded
from that corpus. The author first examined f57v after acquiring a
facsimile of MS 408, underscoring the importance of working from the
physical manuscript pages rather than solely from EVA transcriptions.
The page provides manuscript-internal corroboration of the statistical
analysis: two paths, one
computational (from running text) and one palaeographic (from a diagram),
converge on the same architecture.

Ring 3 enumerates exactly the 13 characters that form the grammar\'s
structural frame (the skeleton characters: o, l, d, r, v, x, k, m, f, p,
t, n, y) and excludes exactly the 6 that fill cores and suffix bodies
(the dressing characters --- so called because they dress the structural frame with vowels and connective elements: e, a, i, c, h, s). The f/p alternation at
position 9 demonstrates gallows substitution at exactly the position the
grammar predicts. The probability of this match arising by chance is p
\< 10⁻⁸ (hypergeometric test for drawing exactly 13 skeleton characters
from 25 EVA characters: p = 1.9 × 10⁻⁷; combined with the specificity
that f/p alternation falls at the gallows slot rather than any of the
four PGCS slots × nine gallows types = 36 slot-type combinations: p =
1/36). Positional tests treating each sequence position as an
independent draw yield p \< 10⁻¹⁶ (here \'exact ordering\' refers to the
slot-sequence ordering --- Prefix precedes Gallows precedes Core
precedes Suffix --- not alphabetic or character ordering). Alternative
readings of f57v have been proposed, including calendrical,
astrological, and decorative interpretations. The statistical match to
the PGCS skeleton characters specifically, not to a generic pattern but
to the exact character inventory that the corpus-derived grammar
predicts in the exact ordering that the grammar requires, renders these
alternatives highly improbable under the PGCS grammar. (Note: the
sequence as described above follows Stolfi\'s H transcription, which
reads positions 7--8 as k-k in units 1, 3, and 4 and k-m in unit 2,
yielding 13 unique characters including terminal *n*. The ZLZI
transcription reads position 7 as *m* in all four units and uses *c*
separators between units, yielding 12 skeleton characters --- all
skeleton, zero dressing --- with the f/p alternation preserved. Under
ZLZI, the hypergeometric becomes p = 2.5 × 10⁻⁶, and combined with f/p
specificity gives p \< 10⁻⁷. The skeleton/dressing partition and the
gallows-substitution finding are robust across all three full-coverage
transcriptions; the k/m alternation is specific to the H reading.)

Ring 1\'s listing of *aiin* and *ar* as atomic units alongside single
characters is consistent with the suffix-family structure documented in
§2.2: suffixes function as pre-assembled blocks, not sequences of
independent characters. The scribe\'s inventory distinguishes two levels
of assembly, characters that are primitives (*c* appears as an individual
separator in Ring 3, distinct from the compound *ch*)
and multi-character units that are pre-assembled (*aiin* and *ar* in
Ring 1), exactly the distinction between the closed-class slots (prefix,
gallows, suffix) and the open-class slot (core) that the grammar
formalises.

Folio 57v also provides independent support for the reclassification of
*ch* and *sh* as prefixes rather than gallows. Ring 3 lists *c* as an
individually spaced character (it appears as a period separator between
the four sequence repetitions) but does not list *ch* or *sh* in the
catalogue. Ring 1 similarly lists *c* as a single character.
Palaeographically, *ch* and *sh* are single glyphs, continuous
pen-strokes, but they share a visible structural relationship with *c*
and *s* respectively. The inventory records base elements from which the
scribe\'s full repertoire can be inferred, consistent with prefix status
(§2.2: 94.1% word-initial dominance; decomposition error 0.001 bits
under PGCS versus ≥1.074 bits under alternative slot assignments).

A generator reading only the f57v transcription, with zero access to the
remaining 225 folios, scores 29 of 84 metrics (enumerated in Supplement
S3). One page, with no corpus fitting and no parameter tuning, already
captures over a third of the manuscript\'s distributional signature. The
metrics it passes are not random: word-length means, entropy, Zipf
parameters, and character-diversity indices all fall within VMS
tolerance, while positional bias metrics (which require corpus-scale
data to calibrate) correctly fail. Folio 57v is sufficient to initiate
production. It is not sufficient to replicate the full statistical
signature, which requires the extended practice that 226 folios of
writing represent.



## S9.2 Three-Layer Detail

Our analysis is built on three layers, each progressively more
constrained. This section introduces three related terms used throughout
the paper: a *quadruple* (or quad) is a complete four-slot decomposition
(prefix, gallows, core-class, suffix-family); a *triple* is the first
three slots only (prefix, gallows, core), which determines how a token
is produced before suffix selection; a *quintuple* adds line position as
a fifth axis.

The first layer is the PGCS architecture: the assertion that words
decompose into four ordered slots with defined inventories. The free
combinatorial product of those inventories yields 656,208 possible
quadruples. The second layer is the constrained grammar (210
rules): the empirical observation that only 5,172 of those quadruples
are attested (0.79%), with 75.4% uniquely determining a single surface
token; this 127-fold compression is what the over-generation hierarchy
(§4.1) quantifies. The third layer conditions the constrained grammar
on section, line position, previous suffix family, paragraph flag, and
quire, expanding 5,172 quadruples to 6,750 quintuples and adding 8
suffix-to-prefix transition distributions; this is what the information
budget (§3) measures, explaining 28.9% of word-selection entropy.

The architecture is a structural framework that could in principle
accommodate many different texts. The constrained grammar is an
empirical inventory of what the VMS actually contains. The contextual
layer specifies when each constrained form appears. Subsequent sections
reference all three levels; the distinction matters because alternative
models (ciphers, constructed languages) may replicate one level without
matching the others.



## S9.3 Within-Word Entropy Detail

The chain rule of entropy (Shannon 1948; Cover and Thomas 2006)
guarantees H(word) = H(P) + H(G\|P) + H(C\|P,G) + H(S\|P,G,C) for any
lossless decomposition. The empirical finding lies in the gap between
this sum and the sum of marginal slot entropies. Unconditional slot
entropies sum to 13.171 bits, exceeding H(word) = 10.311 by 2.860 bits
(21.7% redundancy). PGCS slots are approximately 78% independent (note:
the word-level analysis in this section uses H(word) = 10.311 bits over
character-level tokens; Supplement S2 uses the quad-level H(quad) =
9.124 bits, where each slot is one unit, yielding the 28.9%/71.1%
grammar/lexical partition), with the remaining 22% carried primarily by
the core-suffix association (MI = 0.976 bits, full 2,001-type core),
followed by prefix-core (MI = 0.428 bits) and prefix-gallows (MI = 0.393
bits).

More than half of all tokens (52.7%) have empty cores, composed entirely
from closed-class inventory items. This rate varies by section, from
37.7% (Cosmological) to 63.5% (Balneological). Unlike natural language,
where content words dominate running text, this manuscript\'s vocabulary
is majority-functional. Second-order character entropy (h₂ = 2.13 bits)
nonetheless falls within the natural language range documented by Bowern
and Lindemann (2021), ranging from 1.96 (biological) to 2.23
(pharmaceutical) across sections.



## S9.4 Hapax Origin Decomposition

**Hapax origin decomposition.** The binding constraint is suffix
selection. Decomposition of the manuscript\'s 2,038 hapax types reveals
that 62.2% are variant-hapax: tokens whose prefix-gallows-core triple
appears multiple times in the corpus, but whose specific surface
realisation is unique. Only 37.8% arise from triples appearing exactly
once. The hapax tail is therefore not primarily a product of triple
diversity but of suffix diversity within reused triples.

This suffix diversity is systematic and anti-random. For 27 of the 30
highest-frequency triples, the manuscript produces more distinct surface
types than expected under proportional random sampling from the same
suffix menus; zero triples produce fewer than expected. Across the
corpus, the manuscript contains 629 more hapax than random suffix
selection from attested menus would produce (2,038 observed versus 1,409
expected, averaged over 100 simulations). Context conditioning ---
making suffix choice depend on the preceding token\'s ending --- closes
fewer than 15 of these 629 excess hapax. The mutual information between
suffix choice and preceding token context (NMI = 0.12) is real but
operates in the wrong direction: it narrows the suffix distribution,
making it more predictable given context, rather than driving the rarer
variant selections that produce the excess hapax.



## S9.5 Avoidance Mechanism Detail

**The avoidance pattern is consistent with a production habit operating
over recent writing history, rather than visual scanning of the full
manuscript.** The production evidence in §4.6 shows that 83.9% of tokens
match a nearby source within 10 preceding tokens, consistent with
working from recently written text on the current page or bifolium
spread. A typical VMS page contains 100--210 tokens; the median distance
between consecutive uses of the same triple ranges from 18 tokens
(Balneological) to 32 tokens (Stars), roughly 2--3 lines of text. A
scribe viewing a bifolium spread would therefore see the last 5--15 uses
of any high-frequency triple. If the avoidance were driven by visual
scanning --- \"I can see what I wrote for this entry class, and I choose
something different\" --- a model penalising only the last 5--15 forms
used for each triple should reproduce the effect. It does not.

Per-triple windowed avoidance (windows of 3 to 15 previous uses of each
triple) scores 43--50/84, far below the all-history model\'s 67--76. The
failure mechanism is cycling: when old forms drop off the penalty
window, they become the highest-weighted choice, producing repetitive
oscillation through a small set of forms. The all-history model works
precisely because it never forgets. The scribe\'s avoidance behaviour
operates as though the full production history were accessible, despite
the physical impossibility of scanning 116 folios of previously written
text. This is consistent with an internalised production skill --- a
trained habit of distributing suffix variants, analogous to the way a
trained copyist naturally varies abbreviations or a calligrapher
distributes letterforms --- rather than a conscious visual checking
process. Medieval scribal training produces exactly this kind of
procedural knowledge: motor and cognitive routines for common sequences
that operate below conscious attention. Other behavioural mechanisms
could produce similar statistics, but the windowed model\'s failure
constrains the class of viable explanations to those in which the
effective avoidance memory exceeds the scribe\'s visual range.

Critically, the excess hapax rate is universal: 19.3% ± 1.3% across all
nine manuscript sections (CV = 0.069), from Astronomical (17.9%) to
Stars (21.0%). This uniformity is inconsistent with simple
content-driven explanations for the suffix diversity: if the diversity
arose from section-specific content being notated, sections with
different subject matter would show different diversification rates.
They do not. Uniform diversification could in principle coexist with
structured content if the recording protocol imposes similar constraints
across sections, but the simplest explanation --- that the
diversification rate is a constant of the production method rather than
a reflection of what is being written --- is favoured by the data.

**Section-by-section validation.** The transcription model with
per-triple avoidance was tested independently on all nine sections. Of
the 15 CORE diagnostic metrics, seven pass in every section (9/9):
character-distribution mean, unigram entropy, conditional entropy, mean
word length, unique word length, maximum character frequency, and
digraph coverage. These character-level and word-length properties are
fully determined by the triple stream and suffix inventory; the
avoidance mechanism preserves them. Six metrics pass in 7--8 of 9
sections, with isolated failures attributable to section-specific
properties the model does not address (e.g. triple-selection
autocorrelation in Balneological, hapax spatial clustering in Zodiac).
One metric --- MSTTR, measuring local type-token ratio in 25-word
windows --- fails systematically (4 of 9 sections), and always in the
same direction: the generator over-diversifies locally, producing higher
MSTTR than the manuscript. The VMS scribe suppresses long-range
repetition (avoiding the same surface form across the corpus for a given
triple) while permitting short-range repetition (reuse of the same form
within a few lines). The all-history model penalises all repetition
equally and cannot reproduce this asymmetry.

**The remaining gap.** The residual hapax shortfall (0.56--0.62
generated versus 0.68 observed) concentrates in high-frequency triples:
triples appearing 101+ times show a 42.4% hapax gap, while triples
appearing 2--5 times show only 9.0%. After the binary penalty has been
applied to all available surface forms, it applies equally to everything
and the generator reverts toward proportional sampling. The manuscript
distributes uses more evenly within high-frequency triples than binary
avoidance achieves, but every mechanism tested to force flatter
within-triple distributions (count-scaling, hyperbolic decay,
tau-flattening, windowed lookback) either destroys performance through
cross-triple collisions or introduces cycling artifacts. The remaining
gap resists all tested diversification mechanisms within this model
class.

**Table 5.** CORE-15 diagnostic metrics by section. Model: transcription
with per-triple all-history avoidance. + = pass, X = fail. Scores are
representative single runs; stochastic variation of ±2 points is typical
across random seeds.



## S9.6 CORE-15 Table

| Metric | Stars | Balne | Herb-B | Herb-A | Pharm | Roset | Zodia | Astro | Cosmo | Pass |
|--------|-------|-------|--------|--------|-------|-------|-------|-------|-------|------|
| AC(wordlen) | + | + | + | + | + | + | + | X | + | 8/9 |
| AC(wordfreq) | + | X | + | + | + | + | + | + | + | 8/9 |
| AC(hapax) | + | + | + | + | + | + | X | + | + | 8/9 |
| Char bias mean | + | + | + | + | + | + | + | + | + | 9/9 |
| Char bias skew | X | + | + | + | + | X | + | + | + | 7/9 |
| H₁ (unigram) | + | + | + | + | + | + | + | + | + | 9/9 |
| H₂ (conditional) | + | + | + | + | + | + | + | + | + | 9/9 |
| Word length mean | + | + | + | + | + | + | + | + | + | 9/9 |
| Word length (unique) | + | + | + | + | + | + | + | + | + | 9/9 |
| MSTTR-25 | + | X | X | X | X | + | + | X | + | 4/9 |
| Heaps β | + | + | X | + | + | X | + | + | + | 7/9 |
| Char dist max | + | + | + | + | + | + | + | + | + | 9/9 |
| Digraph coverage | + | + | + | + | + | + | + | + | + | 9/9 |
| Zipf slope | + | + | + | + | X | + | + | + | X | 7/9 |
| Tripled words | + | + | + | + | + | X | + | + | + | 8/9 |
| **Total** | **14** | **13** | **13** | **14** | **13** | **12** | **14** | **13** | **14** | |



## S9.7 Extended Generator Testing

Extended testing beyond the twenty-three generators reported here confirms
that the ceiling is structural rather than parametric. Additional
generator configurations were tested, spanning cache-based
triple selectors with lifetime and eviction dynamics,
context-conditioned suffix models, suffix-novelty mechanisms, and
triple-level length coupling. These tests reveal a Pareto frontier:
slot-assembly generators match type diversity (2,447 types) but
overshoot character-level conditional entropy (H₂ = 2.43 vs VMS 2.17),
while cache-based generators match H₂ (best: 2.08) but collapse the
vocabulary to approximately 1,700 types. No parameterisation of either
family, nor any hybrid, simultaneously satisfies both constraints. The
Stars section\'s operating point (2,982 types, H₂ = 2.17; Table 3) lies
off this frontier.



## S9.8 Full Discussion

**5. Discussion**

**5.1 Engaging Timm\'s Dynamic Hypothesis**

Timm (2026a) argues that VMS text is inherently dynamic, produced
through iterative self-citation, and that static analytical frameworks
fail because they cannot account for a text that was built by copying
and modifying itself. Much of this framing aligns with our findings. The
text is dynamic; section profiles and vocabulary drift across quires
confirm it. Self-citation is real; our generators implement it and the
29 to 46 metrics they reproduce using corpus-wide statistics are
consistent with Gaskell and Bowern\'s experimental finding that
self-citation is the default strategy for producing meaningless text.
Many static analyses do fail to account for the production process.
Where we part company is on sufficiency.

Without the attested quad inventory, no amount of tuning closes the gap
to 59/84. Timm (2026a) acknowledges that the algorithm does not
reproduce certain statistical properties. Our 84-metric suite quantifies
exactly which those are.

The bimodal vocabulary split, once appearing to resist explanation, is a
direct consequence of the PGCS triple structure (§4.2). If self-citation
were the complete production mechanism, hapax legomena should be
accidental copy failures (words that happened not to be reused) and they
should have the same internal structure as frequently copied words. The
97%/61% split shows they do not: hapax types overwhelmingly have filled
cores (97%, type-level), while 61% of non-hapax running text consists of
empty-core tokens. The triple decomposition explains this asymmetry
without requiring two separate production strategies, but the
explanation depends on the grammar. Self-citation alone does not produce
the right internal structure.

A reference page (f57v) is consistent with a taught, transmissible
method; it is not consistent with one scribe\'s gradually evolving
personal habit. The authorship question bears directly on this point.
Fagin Davis (2020) identified five distinct scribal hands through
paleographic analysis. Timm (2026b) subjects this hypothesis to detailed
critique, arguing that the diagnostic criteria (variant forms of k and
n) appear on nearly every page, that the five scribes reduce to
pre-existing Currier A/B categories, and that the handwriting variation
is more parsimoniously explained as continuous evolution of a single
hand. Our distributional analysis offers a different perspective: the
profiles across sections are too discrete to attribute to gradual drift,
with empty-core rates ranging from 37.7% to 63.5% and ED rates from 0%
to 27.9%, while the PGCS grammar holds with zero cross-section
violations. Whether the discrete profiles reflect multiple scribes or a
single scribe working under different conditions at different times, the
structural conclusion is the same: PGCS is a shared, transmissible
method, not an individual habit. The grammar\'s compactness
(approximately 450 table entries) makes it something a scribe could
learn from a colleague or a reference page; f57v is a plausible
candidate for that reference.

Timm and Schinner (2024) also reported temporal artefacts in their
generator output, specifically diachronic patterns that arise
mechanically from the self-citation process itself rather than from any
intentional structure. Timm (2026a) extends this observation to the
manuscript directly, demonstrating that apparent correlations between
text and illustrations are temporal artefacts of batch production: the
scribe(s) wrote all pages sharing the same illustration type
consecutively, and the two Herbal sections, which share the same
illustration type but were written at different evolutionary stages,
exhibit dramatically different vocabularies. This is an important
caution that applies equally to our generators. Any analysis claiming to
find meaningful temporal or sequential patterns in VMS text must first
exclude the possibility that those patterns are artefacts of the
production mechanism. Our information budget (§3) addresses this by
conditioning on sequential context explicitly.

**5.2 Engaging Bowern and Gaskell**

Gaskell and Bowern (2022) established the methodological framework
against which VMS production hypotheses should be tested, and their
experimental data provides important direct evidence about how
humans produce meaningless text. Our generator hierarchy is the
systematic follow-up they proposed.

Their sole unreplicated metric — the tendency of certain characters to appear preferentially at specific positions within words — corresponds to the
PGCS grammar we formalise here. It is not typographic but morphological,
with 210 falsifiable rules and zero held-out violations. Their gibberish
samples were also too short to test section-level structure, and they
flagged this as \"a serious challenge to proponents of the hoax
hypothesis.\" Our section analysis addresses this directly: nine
sections with shared grammar and discrete distributional profiles. And
their classifier identified VMS as more closely resembling gibberish
than meaningful text, a finding our results are consistent with. The
VMS\'s positive AC, low conditional entropy, and high repetition all
fall on the gibberish side.

The two-question framing clarifies the relationship between our work and
theirs. The BG-36 metrics ask: is this gibberish or meaningful text? On
those metrics, grammar makes no difference (generators score 8 to
24 of 36), confirming BG\'s conclusion. The Extended-48 metrics ask a
different question: does this text have character-level morphological
structure? On those metrics, grammar separates generators clearly. The
BG-36 answered their question correctly. The Extended-48 answers a
question they did not ask, and could not have asked, because the
structural metrics require the PGCS decomposition that their work helped
motivate.

The PGCS grammar also explains a property of the VMS that any
decipherment proposal must contend with. The slot constraints are
positional and combinatorial, not semantic: they specify where characters
may appear and which characters may co-occur, not what those characters
mean. Researchers have proposed plaintext mappings from Latin and
Italian (Greshko 2025), Turkish (Ardıç 2025), Chinese (Stolfi
2005), Nahuatl (Tucker and Talbert 2014), and Hebrew, none of which
PGCS rules exclude. The language-agnosticism follows from the grammar\'s
formal structure: a mapping from any
natural-language lexicon to PGCS-compliant tokens can in principle
be constructed without violating the grammar, because the constraints
govern slot occupancy, not word meaning. Every word-level decipherment
is therefore structurally unconstrained by these rules alone.
This does not make decipherment logically impossible --- it
means character-level and morphological structure alone cannot select
among candidate plaintexts. Five centuries of failed decipherment are
consistent with this structural property.

**5.3 Engaging Greshko**

Greshko (2025) approached the VMS from the cipher side, constructing a
historically plausible verbose homophonic substitution, the "Naibbe
cipher", which encrypts Latin and Italian into Voynich-like ciphertext.
His cipher uses an expanded version of the Zattera slot grammar, which
is structurally equivalent to our PGCS decomposition. His ciphertexts
match VMS on many statistics simultaneously, but generate tokens
covering only 45% of Voynich B\'s unique word types (Greshko 2025, §4.1)
and fail on the position-frequency gradient and gallows selection
grammar. Our grammar-enabled generators reproduce the large majority of
attested VMS word types while over-generating by a factor of 2.5 to
3.1×, producing 19,000 to 24,000 unique types against the manuscript\'s
7,598. Without grammar, type recall drops below 25%. The grammar
captures the space of possible words far more completely than the
cipher; the cipher captures the selectivity more precisely.

Greshko\'s results and ours point to the same thing: the slot grammar is
the structural backbone of VMS text, whether you call it a production
grammar or a cipher structure. Neither his ciphertexts nor our
generators fully replicate VMS, and the residual gaps in both cases say
something about what the slot grammar determines and what it leaves
open.

Where we differ is in what the grammar means. Greshko treats it as an
encryption structure: the grammar determines how plaintext maps to
ciphertext. We treat it as a production grammar that organises text
generation regardless of whether the text carries semantic content.
These readings are not mutually exclusive, and the statistical analyses
neither of us performs can resolve the difference. What both approaches
confirm independently is that the grammar is real and empirically
validated.

Greshko scored his "Naibbe cipher" text against the BG benchmark suite,
where it performed well. The two-question framing (§5.2) suggests a
natural extension: scoring Naibbe output against the Extended-48
structural metrics would test whether the cipher replicates
PGCS-specific structure as well as it replicates gibberish-detection
metrics. The scoring code and metric definitions are public.



## S9.9 Full Circularity Disclosure

**5.4 Circularity Disclosure**

***What the scores measure and what they do not***

The generator hierarchy reported in this paper must be read with an
explicit understanding of what is circular and what is not. The
transcription model (Gen-Avoid, 67--76/84 across sections, mean 71.3)
takes the VMS triple stream as input --- the actual sequence of (prefix,
gallows, core) triples produced by the scribe --- and selects suffix
variants from menus extracted from the same corpus. It scores well
because its parts come from the object it is measured against. The score
quantifies goodness-of-fit; it does not constitute an independent
prediction.

Three specific elements of the transcription model are tautological.
First, the suffix menus are corpus-derived: the model can only produce
surface forms attested in the VMS, so high type recovery (98--99%) is
guaranteed by construction. Second, the section-level parameters
(empty-core rates, character entropy profiles) are measured from the VMS
and fed back as generation settings. Third, the triple stream itself is
the VMS triple stream; the model makes no attempt to predict which
triple appears at which position. Any score achieved by such a model
overstates the explanatory reach of the production rules alone.

***What is not circular***

Four findings survive the circularity objection because they involve
comparison, transfer, or external corroboration rather than
self-reconstruction.

**The avoidance mechanism.** The per-triple all-history avoidance rule
was not derived from the data and fed back. It was tested as one
hypothesis among alternatives --- no avoidance, windowed lookback at K =
3 through 15, blended frequency-avoidance models --- on the same corpus.
Frequency-only suffix selection scores 62/84. Adding avoidance raises
the mean to 71.3/84 (range 67--76 across sections). The approximately
9-point mean difference measures a structural property of the
suffix-selection process that no alternative tested can replicate. The
windowed models (43--50/84) fail specifically because avoidance memory
exceeds the scribe\'s visual scanning range, a finding that constrains
the class of viable production mechanisms independently of the score
itself.

**The generator hierarchy.** The progression from f57v-only generators
(16--29/84) through corpus-wide statistics (26--43/84) to
constrained-inventory models with section and position conditioning (up
to 59/84, Gen-SP) to the transcription model (mean 71.3/84) measures the
information contributed by each successive layer of structural
knowledge. This ordering would not hold if the grammar were arbitrary:
randomly constructed slot rules do not produce monotonically improving
scores. The hierarchy is informative even though its upper end is
circular, because the gaps between tiers measure real information
content. The approximately 11-point gap between the best generative
model (Gen-SP, 59/84) and the transcription model (mean 71.3/84) is
particularly diagnostic: it quantifies the information carried by the
specific triple sequence --- the portion of word-selection entropy that
the grammar does not determine. Gen-SP builds everything from scratch;
it scores lower precisely because it is doing real prediction.

**Cross-section transfer.** When the transcription model is trained on
eight sections and tested on the held-out ninth, scores range from 67 to
76 across all nine sections (mean 71.3). The grammar transfers without
retraining. This is the standard method for breaking train-test
circularity, and the result is that PGCS generalises across the
manuscript\'s full format range. A residual circularity remains: the
suffix menus are drawn from the VMS corpus, so the inventory tested is
partly the inventory trained on. Cross-section transfer tests whether
the avoidance rule generalises; it does not test whether the menu
generalises to an independent manuscript. A grammar that overfitted to
one section\'s idiosyncrasies would not transfer.

**Independent physical confirmation.** The PGCS character inventory was
derived computationally from corpus statistics without reference to
f57v. The subsequent discovery that Ring 3 of f57v enumerates exactly
the 13 skeleton characters predicted by the grammar, and excludes
exactly the 6 dressing characters, at p \< 10⁻⁷, constitutes independent
confirmation from a physically separate source. This breaks circularity
for the grammar\'s existence, though not for its parameterisation.

***Recommended reading of the scores***

The honest interpretation of the generator hierarchy is as follows. The
generative models (scoring up to 59/84 with Gen-SP) demonstrate what can
be produced from structural knowledge alone, without access to the VMS
triple stream. Their scores measure genuine predictive power. The
transcription model (mean 71.3/84, range 67--76) demonstrates that,
given the triple stream, suffix selection is well-modelled by
frequency-weighted avoidance. Its score measures fit, not prediction.
The 8--17 metrics that the transcription model fails per section
(approximately 13 on average) represent distributional properties that
neither structural knowledge nor the avoidance mechanism can reproduce
--- properties that depend on the specific sequence of choices the
scribe made, which is precisely what we identify as the content layer.

The self-consistency ceiling (81/84) establishes that even the VMS
measured against itself does not achieve a perfect score under bootstrap
resampling, placing an upper bound on what any model could achieve. The
transcription model reaches 88% of this ceiling (mean). Gen-SP reaches
74%. The gap between them is the triple-stream information, and it is
not recoverable from grammar alone.



## S9.10 Full Limitations

**5.6 Limitations**

The PGCS grammar outperforms the 19 alternative decompositions tested in the validation script
(including Stolfi's crust-mantle-core, conventional ch/sh-as-gallows
parses, systematic boundary shifts, and randomised baselines) by roughly
1,000× in decomposition error, but this is a relative comparison; it is
the best available parse, not a proven unique one. PGCS may be a statistical
abstraction derived from co-occurrence patterns, not a claim about
scribal cognition. Cross-transcription validation (§2.1; Supplement S4)
confirms the structural conclusions are transcription-robust, though
analyses in the main text use the ZLZI transcription throughout.

The generator hierarchy tests one mechanism family. Other production
processes (verbose ciphers, constructed languages, compositional
systems) could be tested against the same metric suite. Greshko (2025)
has begun this work from the cipher side.

The Pareto frontier between type diversity and conditional entropy
(§4.5) is demonstrated within the explored generator class, not proven
for all conceivable mechanisms. A production process operating on units
other than PGCS triples, or incorporating constraints not yet modelled,
could in principle cross the frontier. The per-triple avoidance
mechanism scores 67--76/84 only with access to the exact VMS triple
stream; no fully generative model exceeds 59/84. The anti-random suffix
diversity is confirmed as a production signature inconsistent with
simple content-driven explanations (CV = 0.069 across sections), but the
specific cognitive process underlying the avoidance habit remains
unidentified. The windowed lookback model\'s failure (§4.5) rules out
visual scanning of previously written text as the mechanism, but does
not discriminate between a trained motor habit, an internalised
statistical sense for suffix variation, or a mechanical constraint such
as template cycling.

All significance tests operate on large N (37,465 tokens), so effect
sizes matter more than p-values. Positional effects are small to medium
(Cramér\'s V = 0.09 to 0.11 for suffix-section and prefix-section).
Cross-slot couplings are medium to large (prefix-gallows V = 0.266;
core-suffix V = 0.357, Cohen\'s w = 0.875; core-section V = 0.263). The
word-length autocorrelation is 15.8× its 95% confidence interval under
independence. The bimodal split at type level yields φ = 0.263 with odds
ratio 7.2 (p = 2.3 × 10⁻¹¹⁶). We report mutual information alongside
significance throughout to ensure effects are informationally
substantive.

The bimodal vocabulary finding is descriptive. We report the 97%/61%
split (type-level hapax vs token-weighted non-hapax) as an empirical
constraint, not as evidence for or against meaning. The triple
decomposition (§4.2) explains the split mechanically but does not
resolve whether the filled-core/empty-core distinction reflects semantic
differentiation (content words vs function words), procedural
differentiation (different generation strategies), or some combination.
The finding constrains the production mechanism without resolving the
meaning question.


## S9.11 Extended Future Work

**5.8 Future Work**

The residual gap is now localised to suffix selection within the PGCS
framework, and the production mechanism is partially identified.
Per-triple surface-form avoidance --- a scribe avoiding repetition of
the same surface form for a given triple --- closes 98--99% of the type
gap and approximately half the hapax gap across all nine sections. The
remaining hapax shortfall concentrates in high-frequency triples, where
the avoidance mechanism exhausts its variant space and reverts to
proportional sampling. Three hypotheses were tested to distinguish the
source of the anti-random suffix diversity. First, that it reflects
structured content being notated, predicting section-specific excess
hapax rates. Second, that a scribal production protocol distributes
suffix variants uniformly. Third, that a structural organising principle
imposes diversity at the folio or quire level. The results discriminate
clearly: the excess hapax rate is 19.3% ± 1.3% across all nine sections
(CV = 0.069), inconsistent with simple content-driven explanations for
the diversity. The suffix distributions themselves do vary by section
(cross-section entropy CV = 0.175) and by folio (CV = 0.124), but the
rate of diversification is constant. The combination --- constant
diversity rate, varying suffix distributions --- is consistent with a
scribal production protocol. The 629 excess hapax are better explained
as a production signature than as a reflection of section-specific
content.

The MSTTR failure identifies a specific refinement the current model
lacks. The VMS scribe avoids long-range repetition (the all-history
avoidance that the model reproduces) while permitting short-range
repetition (reuse of the same form within a few lines, which the model
over-penalises). Section-specific local repetition rates vary in a way
consistent with content-type differences: Balneological, with its
formulaic bath diagrams, shows the most local repetition (MSTTR =
0.878); Zodiac, with its varied entries, shows the least (MSTTR =
0.951). A blended model incorporating section-specific local repetition
alongside corpus-wide avoidance improves scores in five sections but
degrades two others, confirming that local repetition is real but that
modelling it requires care to avoid counteracting the avoidance
mechanism.

The distributional properties attributed here to an production habit are
equally consistent with a simpler explanation: that the Beinecke
manuscript is a fair copy of an existing exemplar that already contained
the variant distribution. Under this reading, the all-history model
succeeds not because the scribe remembered their own prior output, but
because the source text already embodied the full variant space, and the
scribe reproduced it with reasonable fidelity. The multi-scribe evidence
(§5.2), the absence of visible corrections, and the copy-mutate
production signature (§4.6) are all consistent with careful
transcription from a legible source. The data do not distinguish between
a deeply trained production habit and faithful copying; both would
produce the observed statistics. This question lies beyond the scope of
the present analysis, but it bears on the manuscript\'s transmission
history.

Two open questions remain. First, what cognitive or mechanical process
underlies the avoidance behaviour --- the windowed lookback model\'s
failure rules out visual scanning of previously written text, but does
not identify the actual mechanism. Second, whether the residual hapax
gap in high-frequency triples reflects a production constraint not yet
modelled or the fundamental limit of binary avoidance applied to triples
with finite variant inventories. The 84-metric scoring framework, the
anti-random suffix diversity test, the per-triple hapax gradient, and
the CORE-15 section-by-section profile provide public benchmarks against
which future proposals can be evaluated.

The bimodal vocabulary split is explained by the triple structure, but
whether the filled-core/empty-core distinction reflects a semantic
content/function-word boundary or a purely procedural asymmetry remains
open. Targeted experiments (generating text under controlled conditions
that vary the ratio of template reuse to novel production) could
distinguish these accounts.

The cross-transcription validation (§2.1) confirms that the four-slot
architecture and information budget are stable across six transcription
systems. Whether finer-grained findings (specific core inventories,
suffix decomposition details, and the over-generation ratios in Table 2)
are equally stable, or whether some are sensitive to transcription-level
glyph boundary decisions that the six-system test does not fully probe,
remains an open question.

Preliminary distributional analysis using PPMI-weighted co-occurrence
embeddings suggests that the four PGCS slots are not equal contributors
to token distribution. The inner slot pair (gallows × core) constrains
distributional behaviour at approximately 5× above chance baseline,
while the outer pair (prefix × suffix) constrains at only 1.2×, with
prefix showing near-zero independent constraining power. Gallows-core
combinations show significant section specificity (χ² test, p \< 0.01)
for 66% of attested groups, covering 96% of all tokens, with each
manuscript section exhibiting a distinctive gallows-core signature.
These findings point toward a two-tier information architecture in which
the inner slots encode content-domain information and the outer slots
encode structural or positional function, to be reported in full
elsewhere.


## S9.12 Section Variation Detail

Timm and Schinner (2024) argued that vocabulary drift under
self-citation could produce ostensible section differentiation: if the
scribe\'s available copying pool changes over time, frequency
distributions will shift. This is true, but pure gradual drift cannot
account for the observed profiles: sections show sharp distributional
breaks rather than smooth transitions. Batch production or
source-switching could also produce discontinuities, but the magnitude
of the differences --- 43.8% of Balneological vocabulary absent from
Herbal --- goes well beyond what gradual drift predicts. More
importantly, 43.8% of Balneological vocabulary is unique to that section
and absent from Herbal, Pharmaceutical, or Stars. Self-citation from a
drifting pool does not explain why nearly half of a section\'s
vocabulary appears nowhere else in the manuscript.


