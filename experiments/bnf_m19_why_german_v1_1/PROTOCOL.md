# BnF M19 German/Bavarian Mechanism Diagnosis v1.1

Date: 2026-08-09
Status: FROZEN BEFORE DIAGNOSTIC RESULTS
Parent: `experiment/bnf-m19-german-confirm-v1.0-20260809`

## Frozen empirical starting point

The v1.0 fixed M19 map is:

`a=5 b=22 c=6 d=4 e=1 f=16 g=22 h=3 i=10 j=20 k=2 l=12 m=9 n=23 o=1 p=7 q=4 r=24 s=30 t=8 u=0 v=28 x=28 y=5 z=20`

On the 122-folio C10 panel, which was untouched by the v0.9 discovery/fitting stage, the map ranked the modern German LM first on ZLZI, TTLI and VDRB, with margins 0.157735, 0.144796 and 0.164746 nats/letter. Lexical permutation z was 5.650, 5.432 and 5.231. All four deterministic C10 buckets ranked German first. v1.0 verdict: `CONFIRMED FRESH-PANEL GERMAN M19 SIGNAL`.

The decoded text is not readable German. This programme therefore asks *why the statistical preference occurs*; it does not assume German plaintext.

## Primary questions

Q1. Is the result specific to the frozen key, or is German commonly favoured by random legal M19 maps on Voynich?

Q2. Which parts of the BnF numerical architecture drive the preference: the five singleton values, other numerical equivalence classes, word boundaries, or internal character transitions?

Q3. Does a period-appropriate German model reproduce the effect, and if so is the preference specifically Bavarian/Austro-Bavarian rather than generic Germanic/high-German structure?

Q4. Does the fixed key transfer to additional transcription traditions beyond ZLZI/TTLI/VDRB?

Q5. Is the preference specific to the *actual BnF 7342 letter-to-value incidence matrix*, or does it survive randomized incidence architectures with the same gross value multiplicities?

## Historical German corpus

Use the official `Referenzkorpus Frühneuhochdeutsch (ReF), 1350–1650`, Zenodo DOI 10.5281/zenodo.5704572 / current archived record 5793616. Use diplomatic tokens from the CorA XML, with document metadata supplied by ReF. Primary historical window: 14th-century second half through 15th-century second half (metadata `14,2`, `15,1`, `15,2`).

Build document-disjoint character LMs for any dialect label with sufficient material. Planned labels include, if supported by the archive: `nordbairisch`, `mittelbairisch`, `südbairisch`, `bairisch`, `alemannisch`, `schwäbisch`, `ostfränkisch`/`nürnbergisch`, and broader comparison groups. No dialect is to be invented or reassigned manually. If a subgroup has <30,000 normalized training letters or <10,000 held-out letters, report it as underpowered and merge only at the corpus's own broader `language-region` level.

Normalization must match the frozen M19 alphabet: lowercase; j→i, v→u, w→u; retain only `abcdefghiklmnopqrstuxyz`. Both diplomatic and modernized ReF forms may be compared, but diplomatic is primary.

## Tests

### T1 — key-null architecture screen

On C10 ZLZI, generate at least 10,000 random legal M19 maps (19 values represented once, six duplicated, 25 surface symbols total). Score each under the induced numerical Markov models for the same eight language panel. Record:

- frequency with which German ranks first;
- frequency with German-vs-best-other margin >= the frozen map's induced-model margin;
- percentile of the frozen map's German score and margin;
- duplicate-value pattern frequency.

This is a fast architecture-bias screen. It is not substituted for the exact forward-HMM result.

For a deterministic stratified sample of at least 64 null maps spanning the induced-margin distribution, run the exact forward HMM on C10. Report empirical exact-forward rank/margin nulls. If compute is prohibitive, report the achieved N and do not extrapolate a p-value below 1/(N+1).

### T2 — anchor/equivalence-class ablation

The BnF incidence matrix has singleton plaintext anchors at values:

- 0 → y
- 22 → o
- 23 → n
- 28 → f
- 30 → s

Under the frozen key these correspond to specific Voynich cipher symbols. For each singleton value, and then for all singleton values jointly, split words at positions carrying the ablated value and rescore the surviving fragments under all language models. Also perform leave-one-cipher-symbol-out and leave-one-value-class-out analyses. Report change in German-vs-runner-up margin and total scored letters.

Do not treat an ablation that removes >25% of letters as clean evidence; label it high-impact/low-coverage.

### T3 — boundary/transition decomposition

Evaluate the frozen key under:

1. ordinary word boundaries;
2. word-start/end terms removed while retaining within-word transitions;
3. unigram-only numerical emissions/frequencies;
4. internal bigram transitions only;
5. reversed-within-word order.

Decompose the induced German-vs-French and German-vs-Spanish log-score difference into start, internal-transition, end, and homophone/frequency terms. Identify the top numerical bigrams and Voynich cipher bigrams contributing to the German advantage.

### T4 — historical/dialect panel

First qualify dialect discrimination on document-disjoint ReF held-out text. A dialect-level result is admissible only if its model can distinguish its own held-out documents above the broader German comparison at a useful rate; otherwise report the dialect resolution as underpowered.

Then apply the *unchanged v1.0 M19 key* to C10 and rank:

- modern German GSD;
- all-period ReF Early New High German aggregate;
- 1350–1500 ReF aggregate;
- qualified ReF dialect models, especially north-, middle- and south-Bavarian;
- qualified non-Bavarian Upper German and Middle German comparators.

No key refitting by dialect in the primary test. Dialect-specific refits may be exploratory only and must be labelled post hoc.

### T5 — additional transcription transfer

With the key frozen, evaluate C10 on all slim-JSON transcription IDs with enough coverage. Primary additional surfaces are `GCGA`, `GCGI`, `TTVE`, `TTIA`, `TTII`, `VDRB-1`, `ZLZB`, `FFSG`, `RGVN`, `PCCA`, subject to mapped-character coverage.

For each surface report coverage, number of folios, modern-language rank/margin, and historical German/dialect rank where evaluable. Do not infer failure from a surface with <90% mapped-character coverage; report it as transliteration-incompatible.

### T6 — BnF-incidence specificity

Randomize the association between the 23 plaintext letters and the 23 BnF five-value profiles while preserving the exact multiset of profiles and all numerical values. Under the frozen Voynich key, compute induced numerical-language scores for at least 10,000 profile permutations. Record how often a randomized incidence architecture yields a German advantage at least as large as the real BnF architecture.

A smaller exact-forward subset may be run if tractable. This test asks whether the result depends on the actual BnF letter/value assignments rather than only on the 19-value channel size.

### T7 — section and Currier diagnostics

Using the fixed key, report German margins by frozen manuscript content section and Currier class. This is explanatory only; it does not create new keys. Compare with the already-passed four random C10 buckets.

### T8 — decoded-output sanity checks

Only after T1–T7 are computed, inspect decoded strings. Measure:

- word-length-conditioned dictionary-hit rates;
- hits against modern German vocabulary versus ReF aggregate and Bavarian vocabularies;
- proportion of hits of length 1–2, 3, 4, >=5;
- common decoded suffixes/prefixes and whether they are model-induced;
- repeated Voynich token → repeated decoded-output consistency.

Dictionary hits of length <=2 must be reported separately and never counted as substantive lexical evidence.

## Interpretation ladder

`ARCHITECTURE BIAS`: random legal keys or randomized BnF incidence reproduce the German advantage frequently.

`GERMANIC PHONOTACTIC SIGNAL`: frozen key is exceptional, historical German reproduces the effect, but Bavarian submodels do not distinguish themselves or lexical/morphological output remains generic.

`BAVARIAN/UPPER-GERMAN SIGNAL`: period-appropriate Bavarian/Upper-German models outperform non-Bavarian historical German on fixed-key held-out data and pass dialect positive controls.

`PLAINTEXT-CANDIDATE SIGNAL`: requires all of the above plus coherent multiword morphology/lexicon beyond short-word effects and successful reconstruction on independent transcription surfaces. v1.1 is not expected to establish this merely from ranking scores.

## Compute discipline

Use CPU unless a demonstrable vectorized/GPU implementation materially reduces cost. Cancel failed/stalled jobs immediately. At programme close, verify no HF jobs remain running.
