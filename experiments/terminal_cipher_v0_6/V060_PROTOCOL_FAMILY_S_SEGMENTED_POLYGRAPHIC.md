# v0.6 Family S — ambiguous segmentation and polygraphic substitution

Date: 2026-07-16

Status: **FROZEN BEFORE IMPLEMENTATION OR RESULTS**

No test data or Voynich text has been inspected.

## Scientific target

Family S covers the remaining historical mechanism not represented by ordinary character substitution or whole-word nomenclators:

- plaintext units may be letters, common digraphs or common trigraphs;
- each unit receives one fresh opaque code group;
- code groups contain one to three visible digits/glyphs;
- groups are concatenated without separators;
- the code set is deliberately non-prefix-free, so boundaries are ambiguous.

This is not a second nomenclator programme. Whole-word codes have already been tested and closed. Family S asks whether changing the underlying unit granularity, then removing code-group boundaries, creates a materially different recoverable channel.

## Literature provenance

The implementation must include the two principal current approaches:

1. BPE and unigram-LM segmentation following Aldarrab & May, *Segmenting Numerical Substitution Ciphers* (2022), arXiv:2205.12527;
2. decoding-lattice search with a neural language model following Chu, Valenti & Knight, *Solving Historical Dictionary Codes with a Neural Language Model* (2020), arXiv:2010.04746.

A recurrence-encoded neural arm may be used as the single permitted development amendment, but shared-code-pool performance must be reported separately and cannot qualify as fresh-key recovery.

## Generator

For each language and split:

- plaintext source length: 384 characters before unitisation;
- candidate polygraphic inventory derived only from the train split:
  - all single characters;
  - top 16 non-space digraphs;
  - top 8 non-space trigraphs;
- plaintext is unitised by deterministic longest match, with single characters as fallback;
- each active unit receives one unique random code string of length 1–3 over a ten-symbol visible alphabet;
- code-length mixture: 20% length 1, 45% length 2, 35% length 3, adjusted only as required for uniqueness;
- prefix collisions are retained rather than repaired;
- the emitted stream contains no separators, word boundaries or unit markers;
- a final joint surface permutation relabels the ten visible symbols;
- fresh codebooks are disjoint across train, development and test trials.

The test split uses independently sampled codebooks and a disjoint set of source chunks.

## Stage S1 — segmentation oracle

Supply the true unit-to-plaintext codebook but hide all code-group boundaries.

Methods:

- exact dynamic-programming lattice using all code strings in the supplied codebook and the train-only character LM;
- BPE and unigram segmentation baselines.

Metrics:

- boundary precision, recall and F1;
- exact unit sequence;
- recovered plaintext character accuracy.

Gate across 16 English development trials:

- mean boundary F1 at least 95%;
- minimum boundary F1 at least 85%;
- mean plaintext recovery at least 95%;
- at least 14/16 trials recover at least 90% plaintext.

Failure closes Family S because segmentation is not identifiable even with the key.

## Stage S2 — segmented polygraphic oracle

Supply true code-group boundaries but hide the fresh mapping from code groups to plaintext units.

Methods:

- character-LM lattice search over candidate plaintext units;
- neural-LM lattice rescoring;
- MDL penalty for unused or unnecessarily long plaintext units.

Gate across the same 16 English development plaintexts with independent keys:

- mean plaintext recovery at least 80%;
- median at least 90%;
- at least 14/16 trials recover at least 80%;
- exact mapping of observed code groups at least 75% on average.

Failure blocks the joint stage. This is required because adding uncertain segmentation cannot rescue an unreliable segmented decoder.

## Stage S3 — fully blind joint segmentation and decoding

Permitted only if S1 and S2 both pass.

The solver receives only the unsegmented visible-symbol stream, observed line boundaries and the frozen train corpus. It must return:

- code-group boundaries;
- plaintext-unit assignments;
- plaintext character sequence;
- calibrated evidence against matched generated controls.

Development gate:

- mean plaintext recovery at least 75%;
- median at least 85%;
- at least 13/16 trials recover at least 75%;
- mean boundary F1 at least 85%;
- no catastrophic trial below 40% plaintext recovery.

One development-only amendment is permitted. It may change search architecture, beam width, neural capacity or training scale, but not the generator, corpus, split, gates or candidate unit inventory.

## Locked test

A passing joint solver is frozen and evaluated once on 20 untouched test trials. No post-test modification is permitted. Voynich application is forbidden unless the locked test passes the same recovery thresholds.

## Dominance rule

Variable-length homophonic character codes without polygraphic units are not rerun as a separate joint family. Segmented fresh-key homophonic recovery already failed its locked reliability gate in v0.5.3; adding unknown boundaries is a strictly harder observation model. Family S therefore tests only the materially new polygraphic-unit hypothesis.