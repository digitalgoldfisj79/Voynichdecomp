# Yiddish M0 necessary-condition invariant programme v0.1

Date frozen: 2026-09-12
Branch: `gpt56/yiddish-m0-invariants-v01-20260912`
Parent scientific state: R finite-panel transfer-short pass; L evaluator closed unresolved after two candidates; T not run; Voynich target sealed.

## Controlling scope

This is a new **control-only implementation/feasibility programme**. It does not rescue or replace the failed L evaluator, does not qualify language discrimination, and does not authorise a Voynich target score. The Voynich transcription must not be loaded by code in this stage.

Question: can we construct measurements that are mathematically invariant under the registered M0 family (one global bijection on orthographic atoms with word boundaries and order preserved), yet are demonstrably non-degenerate under transformations M0 does **not** permit?

A later target addendum is forbidden until this stage closes and the corpus/representation transfer problem is reviewed separately.

## Source controls

### Primary historical-Yiddish source

Columbia MS Gen. 262, repository `cu-mkp/ms-262-data`, frozen commit `1e7b9f78ae1b4d2bc3d6c2c593d0d14855bc664f`.

Use only transcription files `tc_p001r.xml` through `tc_p021v.xml`, and only outer `<ab language="owy">` passages. This range was previously quantity-censused and is therefore **not fresh confirmation evidence**. Its role here is implementation/development of a primary Hebrew-script representation.

Primary extraction `D0`:
- preserve source order;
- recurse through nested language tags inside an outer OWY passage rather than silently deleting borrowings/code-switches;
- `<lb>` becomes whitespace;
- editor-supplied `<exp>` content is omitted from D0 because it is not an explicitly transcribed source atom;
- words are whitespace-delimited after markup removal;
- atoms are raw Unicode code points whose Unicode general category begins `L` (Letter) or `M` (Mark);
- **no Unicode normalisation**, phonetic normalisation, vowel removal, repeat collapse, OTHER bucket, or character transliteration;
- words containing no declared atoms are skipped and counted.

Registered sensitivity `D1` is identical except that `<exp>` contents are included. D1 can expose editorial-expansion sensitivity but cannot rescue a D0 failure.

Nested `<heb>`, `<jit>`, `<lad>`, `<lat>`, `<arm>`, `<laz>` and other marked spans are inventoried separately. They remain in the D0 surface because they are embedded in an OWY discourse; their prevalence is reported as a transfer limitation, not hidden.

### Secondary representation sensitivity

Penn Parsed Corpus of Historical Yiddish frozen at `b5864bd02a315c1d436a82553667bbf81eab6537`, using the already-registered `a-z` word extraction. Penn is consumed and orthographically lossy; it is development/sensitivity material only and never upgrades the primary-script certificate.

## Registered M0 invariants

For a word sequence W, construct both an exact canonical representative and a summary vector.

### Exact canonical structures

1. `canonical_symbol_stream_hash`: concatenate words with a boundary sentinel, rename each distinct atom by order of first occurrence, and hash the resulting integer stream. Under a global atom bijection this must match bit-for-bit.
2. `word_lengths`: full word-length sequence.
3. `token_equality_partition`: replace each distinct whole word by its first-occurrence integer ID.
4. `within_word_patterns`: for every word, replace atoms by first-occurrence IDs local to that word (e.g. ABBA -> 0,1,1,0).

These are implementation witnesses: a planted M0 permutation must leave all four exactly unchanged.

### Summary vector for later compatibility work

- alphabet cardinality;
- sorted atom-frequency spectrum and atom collision probability;
- atom entropy in bits/atom;
- word-length histogram 1..12 and 13+; mean and population SD;
- whole-token type count, type/token ratio, hapax-token fraction, singleton-type fraction, top-1/top-5/top-10 token mass, token entropy;
- fraction of words containing repeated atoms;
- adjacent-identical-atom fraction;
- within-word equality-pattern histogram for the 64 most frequent patterns plus OTHER (ranking frozen from BUILD window only when a later predictive use is specified; this stage instead hashes the complete histogram);
- repeated-token recurrence-gap histogram: 1, 2, 3-4, 5-8, 9-16, 17-32, 33-64, 65-128, 129+; median repeated-token gap.

All quantities must be finite or explicitly `NA`; no nonfinite value is coerced into evidence.

## Fixed windows

Primary D0 uses non-overlapping 512-word windows starting at word 0. The number of available complete windows is data-determined and reported. No window is treated as an independent historical work. Long-window feasibility is separately reported for non-overlapping 2,048-word windows.

The earlier census predicted >=4,128 words, but this is a check, not an assumed fact. If the frozen extractor yields <4,128 words, record the discrepancy and do not change extraction to recover the expected count.

## Planted M0 positive controls

For each complete 512-word D0 window, run 32 independently seeded random global bijections over that window's observed atom inventory. Domain-separated SHA-256 seeds use `(programme, source_commit, window_id, replicate, role)`.

**Implementation pass:** every planted M0 trial must preserve all four exact canonical structures and the complete summary vector exactly. One mismatch => `IMPLEMENTATION_INVALID`; stop before any downstream compatibility analysis.

## Registered non-M0 destructive controls

For each window and 32 deterministic replicates:

A. `WORD_ORDER_SHUFFLE`: permute whole words, preserving the word multiset and each word internally. This should leave length/type/equality marginals intact but alter order/recurrence structure.

B. `WITHIN_WORD_SHUFFLE`: independently permute atoms inside every word, preserving word lengths and global atom multiset. This should alter positional equality patterns/adjacency except in chance-fixed cases.

C. `BOUNDARY_PERTURB`: on a frozen 10% of eligible neighbouring boundaries, merge adjacent words; on a disjoint frozen 10% of words of length >=4, split at an interior position. Atom order is retained but segmentation is not.

D. `SYMBOL_MERGE`: merge the two most frequent distinct atoms into one label (ties lexically by codepoint); no stochastic choice. This is deliberately non-bijective.

These are not labelled 'noncipher'. They test only the specific invariants broken by the registered operation.

**Nondegeneracy gate:** for each destructive family, at least 29/32 replicates in every eligible 512-word window must change the complete summary vector OR the exact canonical structures in the direction that family is capable of changing. If a family has no eligible operation, abstain for that family/window and report it. Failure => `MEASUREMENT_DEGENERATE_FOR_<family>` and no target use.

## Representation sensitivity

Repeat D0 vs D1 on identical primary-source windows. Report exact canonical equality and component-wise summary differences. No choice between D0/D1 is made from whichever is closer to any external target.

Penn representation is assessed only descriptively on already-consumed historical-Yiddish works. Any apparent difference from MS262 is a representation/source/genre mixture and cannot be interpreted as a language effect.

## Fixtures

The implementation must pass:
- empty input -> explicit abstention;
- one-symbol repeated text -> finite defined components where mathematically defined, `NA` elsewhere;
- all-unique word sequence -> recurrence fields explicitly `NA`, not zero or infinity;
- exact encoder/decoder round-trip for every planted permutation;
- seed-domain separation assertions.

## Decision state after this stage

Possible outputs only:
- `IMPLEMENTATION_INVALID`
- `M0_INVARIANT_INSTRUMENT_DEGENERATE`
- `M0_INVARIANT_INSTRUMENT_IMPLEMENTATION_PASS__SOURCE_TRANSFER_UNASSESSED`

Even a pass does **not** unseal Voynich. A later addendum must first state what primary-script Yiddish source population is represented, what source dependence remains, what target transcription/atomisation is used, and what proposition a compatibility or incompatibility result can actually falsify.
