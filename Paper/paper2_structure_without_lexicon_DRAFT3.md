# Structure Without Lexicon: A Convergence Argument Against Recoverable Content in the Voynich Manuscript

Edward Bozzard

*Working Draft — February 2026*

## Abstract

Three independent research programmes converge on the interpretation that the Voynich Manuscript's text was produced by a constrained generative process. Timm and Schinner (2020, 2024) and Timm (2026) demonstrate that self-citation produces VMS-like statistical properties without any source language. Gaskell and Bowern (2022) show, using a random forest classifier trained on 90 metrics across a 932-document corpus, that the manuscript clusters with human-produced gibberish rather than with natural language. Bozzard (2026) identifies the four-slot PGCS grammar (Prefix + Gallows + Core + Suffix) that constrains self-citation into well-formed tokens, with zero held-out violations over 3,930 unseen types.

This convergence closes three doors simultaneously. First, if the text is generated, it is not a cipher, because generation and encryption are structurally distinct. Second, the PGCS grammar is compatible with every natural language tested against it: every cipher mapping is unfalsifiable because the grammar accommodates every source language. No mapping has been verified in over a century. Third, should one reject generation entirely, the cipher alternative requires a compression system without medieval parallel — approximately 29× the information density of a comparable fifteenth-century pharmaceutical manuscript, a 127:1 reduction from combinatorial space, with 61% of running text missing the slot assigned to specific content — whose output clusters with gibberish on every measured dimension. Accept the premise, and lines one and two follow. Reject it, and line three closes behind you. The Voynich Manuscript has grammar, structure, and no recoverable semantic content.

**Keywords:** Voynich Manuscript, convergence, self-citation, PGCS grammar, text generation, cipher falsification

## 1. Three Programmes, One Result

Timm and Schinner (2020) asked whether a generative process could produce VMS-like text. Their self-citation algorithm (iterative copy-modify from previously written tokens) generates Zipf-compliant frequency distributions and positional clustering without any source language. This is an existence proof: no plaintext is required. Timm and Schinner (2024) extended the model, and Timm (2026) makes the strongest claim yet: self-citation explains the text fully, and static analytical frameworks are invalid because the text is inherently dynamic.

Gaskell and Bowern (2022) asked a different question: does VMS text behave like language or like gibberish? Their random forest classifier, trained on 90 metrics extracted from a 932-document corpus including 42 experimentally elicited gibberish samples, placed the Voynich Manuscript on the gibberish side. The strongest discriminating features were character-position bias, word-position bias, compression, repeated words, and word-length autocorrelation. One metric their gibberish samples could not replicate, character bias within words (charbias_words_mean), they attributed to "typographic considerations which cannot be tested rigorously using texts restricted to the lowercase Latin alphabet."

Bozzard (2026) answered what that metric actually is. It is the PGCS slot grammar: 210 rules governing character co-occurrence within four ordered morphological slots, covering 92.96% of 37,465 tokens, producing zero errors on 40% held-out data. When the grammar is incorporated into text generators, output moves toward the gibberish regime identified by Gaskell and Bowern, not toward the language regime. The jump from no-grammar generators (37–50 of 90 metrics) to grammar-enabled generators (58–67) is categorical, not incremental.

No single strand proves the case. Timm demonstrates that generation is possible. Gaskell and Bowern show that VMS behaves like generated text. The PGCS grammar explains how generation produces well-formed output rather than degenerating noise. Together, these three results converge on the generative interpretation. The question is whether any cipher model survives as an alternative. Sections 2–4 test whether it can.

## 2. Generated Text Is Not Encrypted Text

A cipher transforms an existing message. Even elaborate polyalphabetic and homophonic systems preserve statistical traces of their source language: word boundaries, mutual information between tokens, distributional footprints. This is why cryptanalysis works. A generative process produces tokens directly. Tokens are the primary objects, not transformed representations of earlier linguistic units.

Six features of VMS text either contradict specific predictions of the cipher hypothesis or are predicted by the generative interpretation.

Self-citation produces VMS-like distributions without any source language (Timm and Schinner 2020, 2024; Timm 2026). A cipher predicts that some source language is necessary; none is required. The PGCS grammar constrains token formation without reference to any semantic content (Bozzard 2026, §2–3). A cipher predicts that structural constraints reflect source-language morphology; the PGCS constraints are formal and language-independent. The grammar is compatible with every natural language tested against it (Bozzard 2026, §5), the mirror property. A cipher predicts that pattern-matching to candidate source languages can discriminate between them; it cannot. Adding structural sophistication to generators moves their output toward gibberish, not toward language (Bozzard 2026, §4). A cipher predicts the opposite: more structure should produce more language-like output. Five scribes follow the same grammar with different statistical profiles, consistent with a shared production method rather than five independent encipherments (Bozzard 2026, §4; Fagin Davis 2020). The text exhibits no word-sequence grammar beyond suffix-to-prefix character coupling across token boundaries (Bozzard 2026, §3.3). A cipher predicts that word-order patterns from the source language survive encryption: recipe formulae, prepositional chains, verb-argument structure — the syntactic scaffolding present in every known pharmaceutical manuscript of the period. Homophonic substitution changes tokens, not their sequence. The VMS preserves sub-word structure (PGCS) without preserving word-order structure, exactly as self-citation from visual proximity predicts.

The mirror property deserves emphasis because it resolves a long-standing puzzle. Researchers have found plausible statistical matches to Latin (Greshko 2025), Turkish (Keskıntoğlu, Türkmen, and Ahat 2019), Chinese (Stolfi 2005), Nahuatl (Tucker and Talbert 2014), and Hebrew, among others. Each match is genuine in the sense that the grammar accommodates it. None is verifiable because the grammar accommodates them all. The constraints are formal, not semantic; they restrict the shape of words without specifying what the words mean. Any language held up to the grammar reflects back as a plausible source.

### 2.1 Greshko's Naibbe Cipher

The strongest recent case for the cipher hypothesis is the Naibbe cipher (Greshko 2025): a verbose homophonic substitution system, executable by hand with fifteenth-century materials, that encrypts Latin and Italian into VMS-like ciphertext. The Naibbe cipher is important because it demonstrates that the cipher hypothesis is not ruled out by historical implausibility. A workable mechanism exists.

It also illustrates the limits of that hypothesis. The cipher reproduces approximately 30% of unique Voynich B word types and 83% of total tokens. Its output is classified as gibberish at low confidence by Bowern's classifier, exactly as VMS is. And it operates on an expanded version of the slot grammar observed in the manuscript, confirming the mirror property from a direction Greshko did not intend: the grammar accommodates a cipher interpretation just as readily as a generative one.

The Naibbe cipher demonstrates that a workable cipher mechanism exists. The question is not whether a mechanism could produce VMS-like output but whether the VMS text bears the traces of one having been used. The six features above and the empirical results in §3 show it does not.

## 3. No Recoverable Content

Four empirical results from Bozzard (2026) test whether the text preserves any recoverable semantic signal.

A permutation test shuffling illustrations across folios 500 times established a null baseline for word-image correlation. The manuscript's actual correlations were no stronger than chance. The text shows no detectable response to its own illustrations. Timm (2026) reaches the same conclusion independently, showing that apparent text-illustration correlations are temporal artefacts of batch production: the scribe processed illustration types consecutively, creating section-vocabulary associations that reflect writing order, not semantic content.

The 3,285 hapax legomena with filled cores, structurally complex unique tokens that would carry specific content under any semantic interpretation, do not cluster by section or illustration proximity. Their distribution follows Zipf's law and nothing more.

The PGCS grammar accounts for 28.9% of word-selection entropy. The remaining 71.1%, the content layer, is where any hidden language would live. This layer carries no recoverable signal pointing to any specific source language.

The bimodal vocabulary has the structural shape of a function-word/content-word distinction (61% of high-frequency running text has empty cores; 97% of hapax types have filled cores) but not the semantic behaviour. The "content" hapax show no topical clustering. Shape without meaning is what constrained generation predicts.

These results are predicted by the generative interpretation and unexplained by cipher. If the text encodes a plaintext, something should correlate with something: words with images, hapax with topics, the content layer with a source language. Nothing does.

A clarification on what "recoverable" means. This paper does not claim the text is meaningless in some absolute sense. A random number generator with a fixed seed contains information in the sense that the seed determines the output, but no analysis of the output alone recovers the seed. The VMS production process may have had inputs. But the output does not preserve those inputs in recoverable form. The claim is *irrecoverable*, not *absent*.

### 3.1 The Circularity Problem

All cipher models require at least one independently verifiable mapping from token to plaintext. No proposed mapping for the Voynich Manuscript has produced verifiable results in over a century of effort.

This would be suggestive but not conclusive were it simply a failure of effort. What makes the problem structural is the mirror property established in Bozzard (2026). Because the PGCS grammar is compatible with any source language, pattern-matching to a candidate language is unconstrained. A researcher who proposes Latin will find Latin-compatible patterns. A researcher who proposes Turkish will find Turkish-compatible patterns. Both are correct in the narrow sense that the grammar accommodates their hypothesis. Neither can validate their mapping because the grammar cannot discriminate between them. The lock accepts every key.

The circularity does not depend on accepting the generative interpretation. It depends on accepting the PGCS grammar, which is supported by zero held-out violations across 3,930 unseen types (Bozzard 2026). Given that grammar, every cipher mapping is unfalsifiable. This is not a failure of ingenuity. It is a structural property of the system.

The same circularity applies to external reference hypotheses, the proposal that tokens are pointers into a lost codebook. A token-to-index-to-meaning mapping is still a substitution system. The circularity does not care how many layers of indirection exist. Without the codebook, the mapping is unconstrained. Without independent verification of any single mapping, the codebook is unrecoverable. This is unfalsifiable by construction.

## 4. The Density Problem

The cipher hypothesis makes specific quantitative predictions about information density. Those predictions can be tested directly.

If the PGCS slots encode independent semantic content (prefix for plant category, gallows for condition, core for preparation, suffix for application), then each VMS token carries four information units. A page with 100 tokens encodes 400 units. A representative page of Brescia, Biblioteca Civica Queriniana MS B.V.24, a fifteenth-century Lombard pharmaceutical compendium containing fertility remedies, herbal entries, and therapeutic recipes in mixed Latin and Italian vernacular, carries approximately 14 independent information units per 100 tokens — ingredients, preparations, dosages, conditions, and instructions — with the remainder consumed by syntactic scaffolding (*et*, *in*, *de*, *per*, *cum*). Under the cipher interpretation, the VMS would need to be approximately 29 times denser than a real pharmaceutical manuscript of the same period and region.

This requires: a compression system without historical parallel; used in a working notebook, not a prestige production; shared across five scribal hands who each follow it without error; whose output is classified as gibberish by the most rigorous statistical classifier applied to date (Gaskell and Bowern 2022); where the compressed content shows zero correlation with the manuscript's own illustrations; and where unique tokens show no semantic clustering by topic or location.

Two specific features of the PGCS grammar sharpen the difficulty.

The free combinatorial product of the four slots is approximately 4.75 million possible types. The manuscript uses approximately 5,000, a 127-fold compression. If the slots independently encoded different semantic dimensions, they would combine more freely. They do not. The compression demonstrates massive inter-slot dependency: the slots function as a grammar, not as four parallel encoding channels.

61% of high-frequency running text has empty cores (Prefix + Gallows + ∅ + Suffix). Under the semantic interpretation, the core slot is where specific content lives, the preparation method, the active ingredient. It is absent in the majority of running text. The high-frequency stratum is scaffolding without the slot assigned to the most variable content.

This section is not a fallback for readers who reject the preceding arguments. It is the reductio that closes behind them. Reject the generative interpretation and reject the circularity argument, and you require: a cipher producing approximately 29× the information density of a real fifteenth-century pharmaceutical manuscript, at a 127:1 compression from combinatorial space, with 61% of running text missing its content slot, whose output shows no correlation with its own illustrations, no semantic clustering, and no recoverable signal in the content layer — all while clustering with gibberish across every measured dimension. The door closes.

## 5. The Exact Mechanism Is Irrelevant

Once the generative interpretation is accepted, the question "which exact mechanism produced this text?" becomes historical rather than cryptographic.

The generator hierarchy in Bozzard (2026) identifies three metrics no generator matches simultaneously: word-length autocorrelation (AC(1) = +0.160), lexical repetition rate (0.008), and local vocabulary diversity (MATTR-25 = 0.919). Gaskell and Bowern (2022) independently identified two of these three among their classifier's ten strongest discriminating features. This convergence was not designed: the metrics were selected by a random forest classifier and by a generator hierarchy independently. The overlap tells us something real about VMS production, but what it tells us is forensic, not semantic.

These metrics characterise a specific production run. They may reveal something about workshop practice: copying speed, column-scanning patterns, scribal habits, the physical act of looking at nearby text. They cannot resurrect the meaning question. The gap between the best generator and the manuscript tells us that the historical production process involved dynamics no modern simulation has fully replicated. It does not tell us the text means anything. Fingerprints, not DNA.

Where we part company with Timm (2026) is not on generation (he is right that the text is dynamic) but on the relationship between mechanism and grammar. Timm argues that static frameworks fail. This is correct, and the PGCS grammar is what makes it correct: the grammar provides the constraint system within which dynamic self-citation produces well-formed output. Without grammar, self-citation degrades. With it, self-citation produces the right statistical regime. Timm's mechanism and our grammar are complementary. The mechanism explains the process; the grammar explains why the process does not degrade. Together, they yield the production model: constrained self-citation within shared grammar, performed by multiple scribes.

The generator hierarchy served its purpose not by identifying the exact mechanism but by demonstrating that grammar plus self-citation reaches the correct statistical regime, and that further refinement moves output toward gibberish rather than toward language. The hierarchy is a proof of concept for the *class* of process. The specific instance is lost to history.

## 6. What Remains

The remaining question is historical: why did five scribes collaborate to produce a carefully illustrated manuscript of structured text with no recoverable content?

This paper does not answer it. But it reframes the field. The Voynich Manuscript is not a cryptographic puzzle awaiting the correct key. It is a historical artefact whose production method is understood in its general form, and whose purpose must be sought in the circumstances of its creation rather than in the content of its text.

The radiocarbon date of 1404–1438 (Pettit et al. 2009) and the codicological evidence for Northern Italian production constrain the answer to a specific window and region. Whether the manuscript is a workshop exercise, a commissioned object whose value derived from apparent complexity, or something else, that question belongs to book historians, not cryptographers. The code was never there. The structure was.

## 7. Conclusion

Three independent research programmes converge on the generative interpretation. The convergence is not the only argument. It rests on three lines, each of which closes a different escape route.

The first line: if the text is generated, it is not a cipher. Six features of VMS text contradict cipher predictions or are predicted by generation. No source language is required to produce VMS-like distributions. Adding structural sophistication to generators moves output toward gibberish, not toward language. The boundary between cipher-produced and generation-produced text is statistically blurry, but the text bears none of the traces a cipher would leave.

The second line: no cipher mapping is verifiable. The PGCS grammar is compatible with every natural language tested against it, making every proposed mapping unfalsifiable. This circularity is not a failure of effort but a structural property of the grammar. Given language-agnostic constraints, pattern-matching to any candidate source language is unconstrained. The lock accepts every key. Every semantic test applied to date returns null: no word-image correlation, no hapax clustering, no content-layer signal.

The third line: reject generation, and the cipher alternative requires an information density without medieval parallel. Approximately 29× the density of a comparable fifteenth-century pharmaceutical manuscript. A 127:1 compression from combinatorial space. 61% of running text missing the slot assigned to specific content. Zero correlation with illustrations. Output classified as gibberish across every measured dimension.

Accept the generative premise, and lines one and two follow. Reject it, and line three closes behind you. The Voynich Manuscript has grammar, structure, and no recoverable semantic content. Structure without lexicon.

Five centuries of effort have not found meaning in this text. The convergence of independent programmes now explains why.

## Acknowledgements

The convergence argument presented here rests on work by Torsten Timm, Andreas Schinner, Claire Bowern, and Luke Gaskell, whose independent programmes made the synthesis possible. Michael Greshko's Naibbe cipher, by demonstrating the mirror property from the cipher side, sharpened the argument by showing that mechanism existence does not entail mechanism use. Lisa Fagin Davis's paleographic identification of five scribal hands provides the multi-hand evidence central to §2. ChatGPT (GPT-4o) and Claude (Anthropic) were used as computational research assistants for manuscript preparation; all analytical decisions and interpretations are the author's.

## References

Bozzard, E. (2026). Slot Grammar and Self-Citation in the Voynich Manuscript: A Generator Hierarchy. Submitted to *Cryptologia*.

Brescia, Biblioteca Civica Queriniana MS B.V.24. Pharmaceutical compendium, fifteenth century, Lombardy. Mixed Latin and Italian vernacular. Transcription via Transkribus.

Currier, P. (1976). Papers on the Voynich Manuscript. *New Research on the Voynich Manuscript: Proceedings of a Seminar*. Washington, D.C.

d'Imperio, M. E. (1978). *The Voynich Manuscript: An Elegant Enigma*. Fort Meade, MD: National Security Agency.

Fagin Davis, L. (2020). How many glyphs and how many scribes? Digital paleography and the Voynich Manuscript. *Manuscript Studies*, 5(1), 164–180.

Gaskell, D. E. and Bowern, C. (2022). Gibberish after all? Voynichese is statistically similar to human-produced samples of meaningless text. *CEUR Workshop Proceedings*, Vol-3313, International Conference on the Voynich Manuscript 2022. University of Malta.

Greshko, M. A. (2025). The Naibbe cipher: A substitution cipher that encrypts Latin and Italian as Voynich Manuscript-like ciphertext. *Cryptologia*. doi:10.1080/01611194.2025.2566408.

Keskıntoğlu, G., Türkmen, A., and Ahat, M. (2019). [Verify full title and publication details.]

Pettit, A. G. et al. (2009). Radiocarbon dating the Voynich Manuscript. Report for Yale University.

Stolfi, J. (2005). Voynich Manuscript word structure analysis. *Online manuscript*.

Timm, T. and Schinner, A. (2020). A possible generating algorithm of the Voynich manuscript. *Cryptologia*, 44(1), 1–19.

Timm, T. and Schinner, A. (2024). The Voynich manuscript: Discussion of text creation hypotheses. *Cryptologia*, 48(4), 305–322.

Timm, T. (2026). The challenge of analyzing a dynamic text: Why the Voynich Manuscript resists conventional interpretation. *Cryptologia*.

Tucker, A. G. and Talbert, R. H. (2014). A preliminary analysis of the botany, zoology, and mineralogy of the Voynich Manuscript. *HerbalGram*, 100, 70–75.

---

*Draft 3, February 2026. Approximately 3,500 words excluding references.*
