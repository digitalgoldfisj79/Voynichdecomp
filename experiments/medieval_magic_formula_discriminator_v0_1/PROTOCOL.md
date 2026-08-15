# MEDIEVAL MAGIC FORMULA × VOYNICH MECHANISM DISCRIMINATOR
## Frozen protocol v0.1 — 2026-08-15

## Purpose
Test whether historically attested medieval magical/medical textual production can generate the combination of structural properties seen in the Voynich Manuscript, without fitting the mechanism to Voynich. This does not test whether the Voynich Manuscript is a grimoire and does not count lexical coincidences as evidence.

## Hard constraints
1. Fit mechanisms and hyperparameters without Voynich.
2. Keep Voynich sealed until external discrimination and synthetic-recovery gates pass.
3. Voynich representation hierarchy: RF -> STA-family -> full STA -> connected aaa.
4. No post-hoc threshold tuning.
5. Split external data by source witness/formula family, not random line.
6. Held-out Voynich inference is by folio/quire.
7. Synthetic/permutation controls mandatory.
8. No individual lexical coincidence can create a positive.
9. One-representation effects are non-robust.
10. Stop at decisive negatives.

## External classes
A ordinary medieval language: Latin/German medical recipes, religious prose, matched medieval prose.
B corrupted/hybrid formulaic language: distorted Latin prayers, Latin+vernacular mixtures, copied Greek/Hebrew names, abbreviation/fusion/splitting, oral/scribal corruption.
C productive voces magicae/formula families: TERIX/CONTERIX/PETRONIX-type families, EREX/AREX/RYMEX-like families, festella paradigms, Peda/inpeda/prepeda paradigms, repetitive/reductive strings, generated names.
D mixed medical miscellany sequences: document-order recipe/charm/prayer/remedy mixtures, manuscript boundaries preserved.

## Lecouteux extraction fields
entry_id, canonical_headword, exact_formula, normalized_formula, source_work, manuscript_shelfmark, folio/page, date_min, date_max, region, languages, function, medical, christian_anchor, opaque_voces, segmentation, repetition_type, productive_family, corruption_evidence, abbreviation_evidence, cipher_evidence, source_confidence, primary_source_checked.

Lecouteux is a discovery index. Load-bearing historical claims require witness/critical-edition checking. No copyrighted prose descriptions are redistributed; only short formulae, metadata, classifications and numerical features are retained.

## External split
60% development / 20% validation / 20% held-out test, grouped by witness/formula family. No variant of a family may cross splits.

## Feature families
F1 entropy/predictability: H1, conditional entropy orders 1-3, MI lags 1-20, excess-entropy proxies, compression, cross-validated next-unit prediction.
F2 token-family geometry: lengths, type-token, hapax, nearest edit distance, edit graph, prefix/suffix sharing, stem-affix compressibility, one-edit asymmetry, family branching entropy.
F3 repetition/local-copy: repeat distances, near-copy distances, within-line repetition, adjacent similarity, runs, template recurrence, local mutation conditional on previous token.
F4 position: line-initial/final divergences, paragraph-initial, token length by position, prefix/suffix by position, family members by line position.
F5 document conditioning: between-document JSD, section predictability, local-vs-global LM gain, scribe conditioning, mixture purity.
F6 long-range: recurrence, autocorrelation, burstiness, sequence segmentation, change points.
F7 compression-transfer: CT(A->B)=C(A+B)-C(A), symmetric NCD, zlib/bzip2/LZMA mandatory, tokenized/untokenized/symbol-renamed/fixed-width forms.

## Historically constrained generators
G0 ordinary language baseline.
G1 scribal corruption: deletions, duplications, fusions, splits, attested letter confusions, abbreviation errors; rates external only.
G2 oral corruption preserving approximate sound pattern; external variant-family calibration.
G3 formula-family stem + controlled prefix/suffix mutation.
G4 repetition/reduction generator.
G5 hybrid medical/charm switching model with externally estimated switching rates.
G6 abbreviation/segmentation generator.
G7 only source-attested simple cipher controls: shifts, reversal, vowel replacement/numbering, simple substitution, abbreviation coding.
G8 constrained combination: at most one mechanism per independent class; no free stacking unless components qualify.

## Qualification gates before Voynich
Q1 synthetic positive-control recovery: macro-AUC >=0.80 OR top-1 mechanism identification >=70%, bootstrap lower 95% CI > chance+0.10.
Q2 held-out A/B/C discrimination: macro-AUC >=0.75 and each class AUC >=0.65.
Q3 family leakage: no family crosses split; grouped performance >=90% of random-split performance.
Q4 length/alphabet controls: repeat after length matching, alphabet-size matching by symbol renaming, token-frequency permutation, within-line shuffle, between-line shuffle. Signal cannot reduce to alphabet size, length or unigram frequency.
Q5 null calibration: shuffled labels, shuffled document blocks and synthetic Markov controls; FDR q<=0.05 or preregistered max-T.
Only metrics/generators passing Q1-Q5 advance.

## Sealed Voynich primary statistic
For each representation calculate standardized external-space distances to A/B/C/D centroids/posterior predictive distributions.
Delta_magic = d(V,A) - min(d(V,B),d(V,C),d(V,D)).
Standardization is learned externally only.

## Programme-level positive
Requires ALL:
1. Delta_magic >0 on every representation.
2. Same winning external class on >=3/4 representations.
3. Effect exceeds 95th percentile of matched ordinary-language controls.
4. Effect survives held-out folio/quire bootstrap.
5. No single feature family contributes >50% of total discrimination.
6. Nearest magic class is materially closer than synthetic Markov/slot controls.
7. Qualified external generator reproduces >=70% of preregistered Voynich target metrics within its 95% posterior predictive interval.

## Negative / unresolved
Strong negative if no B/C/D robustly beats A, if effect disappears under symbol/length controls, if best generator reproduces <50% of target metrics, or representation agreement <3/4.
50-70% target metric coverage = non-resolving.

## Adversarial controls
N1 phonotactically plausible modern nonsense/glossolalia.
N2 matched Markov model.
N3 delexicalized slot grammar.
N4 scribal corruption only.
N5 formula family only.
N6 mixed unrelated ordinary texts at matched switching rate.

## Secondary tests only after primary non-negative
S1 section-specific affinity; S2 Currier/hand modulation controlling section; S3 line-position mechanism; S4 local formula analogues. Visual/layout extension is separate and cannot contribute to primary p-values.

## Interpretation
A: robust magic-class win + >=70% metric coverage -> historically constrained magical-medical mechanism serious candidate.
B: closer but 50-70% -> structural affinity, insufficient explanation.
C: disappears under controls -> artifact.
D: ordinary prose/Markov as good or better -> reject magical-formula mechanism.
E: only f116v/marginalia show charm affinity -> supports marginal charm, not whole-manuscript mechanism.
