# Source-transfer v0.4.0: oracle identifiability protocol

Status: **frozen before execution**.

Base commit: `0962441a373ca4aa045c19ef56fd010dd926cca9`.

Branch: `experiment/source-transfer-v0.4-oracle-audit-20260715`.

## 1. Scientific boundary

This programme does **not** claim that ciphertext alone permits a universal distinction between enciphered messages and structured generation. The two unrestricted hypothesis classes overlap. The test is conditional on the source-model and generator families represented in the benchmark.

The immediate question is narrower:

> Under oracle latent recovery, does a candidate representation retain source-specific structure, and is that evidence specific to real held-out source sequences rather than ordered or source-conditioned non-message generators?

No Voynich text is scored in v0.4.0.

## 2. Reason for the oracle stage

v0.3.4 combined two possible failure modes:

1. failure to recover the latent mapping;
2. failure of the recovered latent representation to distinguish message-bearing sequences from ordered non-message sequences.

v0.4.0 removes mapping inference entirely. It operates on source-derived representations directly. A representation that fails here cannot be rescued by a better beam search, parallel tempering, neural decoder or larger compute run.

Cipher surface renderers are deliberately omitted from the primary oracle score. Once the true mapping is supplied, renderer choice is mathematically irrelevant except where the renderer destroys segmentation or inserts unlabelled noise. Renderer robustness belongs to the next inferred-mapping stage and will be run only for representations that pass this gate.

## 3. Developmental corpora

The first run uses the two existing independently sourced Greek word streams:

- `Paper/Cipher_paper/greek_corpus_parsed.pkl`;
- `Paper/Cipher_paper/greek_dmm_corpus.pkl`.

Each currently exposes an ordered `all_words` stream but no reliable work-level boundaries. Therefore v0.4.0 is a **developmental falsification test**, not a formal locked validation. Passing requires a later corpus rebuild with work, author, period and genre provenance and work-disjoint splits.

Each corpus is split contiguously without wraparound:

- first 60%: source-model training;
- next 20%: threshold and model-order development;
- final 20%: untouched test.

Exact token counts, corpus hashes and train/test vocabulary overlap are recorded.

## 4. Candidate representations

The following representations are frozen:

1. `word12`: the current 12-class word projection;
2. `initial`: first orthographic/phonemic unit;
3. `char`: complete normalized character stream with word boundaries;
4. `syllable`: deterministic orthographic syllable-like units;
5. `bpe`: pooled-training-only deterministic BPE units;
6. `word`: pooled capped word vocabulary with an unknown-word class.

The BPE tokenizer and capped vocabulary are trained on pooled source-model training partitions only. No development or test token influences tokenization.

## 5. Source models

For every representation and source corpus, train smoothed token n-gram models of orders 1 through 5 using training chunks only.

For a held-out sequence from source `s`, define the source-transfer margin:

`margin = best_wrong_source_bits_per_token - correct_source_bits_per_token`.

Positive margin means the correct source model gives the shorter code.

Model order and the decision threshold are selected on development data only. For each order, choose the threshold giving maximum positive sensitivity subject to development false-positive rate no greater than 10%. Select the order with the greatest constrained sensitivity, breaking ties by lower false-positive rate and then higher order. Freeze this choice before test scoring.

## 6. Non-message controls

For each positive development and test sequence, generate one same-length control from every frozen family:

- `shuffle_keep_first`;
- `block_shuffle`;
- `bigram_euler`, preserving the exact directed bigram multigraph;
- `iid_unigram`, sampled from the source training distribution;
- `source_markov1`;
- `source_markov2`;
- `ordered_hmm`;
- `motif_grammar`;
- `topic_fsm`;
- `copy_mutate`.

The source-conditioned unigram and Markov controls are intentionally hostile. They test the central limitation of source-family transfer: a source-trained generator may retain source attribution without containing an independently selected message.

## 7. Information-preservation diagnostics

For each representation, report:

- train-to-test word decoder accuracy from representation signature;
- rate of test signatures observed in training;
- word entropy;
- conditional word entropy given representation;
- normalized mutual information;
- unique-signature rate;
- number of distinct signatures and word types.

These diagnostics are descriptive. They do not substitute for the held-out message-specificity gate.

## 8. Sample sizes

Default full developmental run:

- 20 development chunks per source;
- 24 test chunks per source;
- 256 source words per chunk;
- 2 real source families;
- 10 control families per positive;
- 6 representations;
- n-gram orders 1–5.

This yields, per representation, 48 real test sequences and 480 test controls before order selection.

The unit of analysis is the independently selected source chunk. Token-level observations are not treated as independent trials.

## 9. Gates

### Source-transfer gate

A representation passes source transfer only if, on untouched test data:

- correct-source rank-1 rate is at least 80%; and
- median source-transfer margin is at least 0.02 bits per token.

### Non-source ordered-specificity gate

For the ordered controls not explicitly sampled from the source unigram/Markov models:

- positive sensitivity must be at least 70%; and
- no control family may have a false-positive rate above 25%.

### Full message-specificity gate

Including all source-conditioned controls:

- sensitivity at least 70%;
- aggregate false-positive rate at most 15%;
- no control family false-positive rate above 25%.

A representation is eligible for inferred-mapping work only if it passes both source transfer and full message specificity.

Failure of all representations stops the present source-transfer architecture before any expensive mapping or neural run. Passing is necessary but not sufficient for a formal multilingual validation.

## 10. Multiplicity and leakage controls

- Representations, model orders, control families, thresholds and gates are frozen here before execution.
- Order and threshold selection use development data only.
- Test results are evaluated once.
- BPE and vocabulary construction use training data only.
- Exact source and code hashes are written to the result artifact.
- No decision threshold may be changed after viewing test results.
- The two-corpus developmental run cannot support language-universal claims.

## 11. Compute design

The programme is optimized for elapsed time rather than minimum cost:

- source models are fitted once per representation;
- test scoring is fork-parallel across CPU workers with copy-on-write model sharing;
- the full gate runs on `cpu-xl` with 32 workers;
- BPE training and classical n-gram scoring are CPU-bound and do not justify GPU startup or GPU cost;
- GPU compute is deferred to inferred neural mapping or multilingual neural source models, and only after this oracle gate passes.

## 12. Interpretation

Passing would establish only that a representation carries held-out source-family evidence that the frozen non-message controls do not reproduce at the accepted rate.

It would not establish that:

- the Voynich Manuscript is enciphered;
- the source language is Greek;
- unrestricted generators are distinguishable from unrestricted ciphers;
- a usable mapping can be inferred from ciphertext.

The next stage, if authorized, must cross the surviving representation with independently implemented cipher renderers, keys, null mechanisms, segmentation changes and generator families, and must evaluate oracle versus inferred mapping separately.
