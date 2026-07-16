# Recoverability frontier v0.5.4 — nomenclator recoverability

Date frozen: 2026-07-16

Status: **development programme; no Voynich inference permitted**

## Motivation

The v0.5.3 fresh-key homophonic bake-off failed its complete locked reliability programme. Per the frozen decision rule, that search line is closed rather than extended through further ad hoc tuning.

v0.5.4 moves to a structurally different and historically relevant family: a monoalphabetic character substitution combined with opaque whole-word nomenclator codes.

## Scientific question

Can a system recover independently selected plaintext when:

- ordinary characters are transformed by a fresh monoalphabetic key;
- selected frequent words are replaced by fresh opaque code symbols;
- character symbols and code symbols are jointly relabelled so numeric ranges reveal no type information;
- the plaintext language and broad cipher family are known, but neither key nor codebook is supplied?

## Generator correction

The original v0.5.0 nomenclator generator placed character symbols below the plaintext alphabet size and code symbols above it. First-occurrence canonicalisation reduced but did not formally eliminate the possibility of implementation-specific type leakage.

v0.5.4 uses a replacement generator:

1. create a fresh monoalphabetic character key;
2. sample a fresh nomenclator codebook from a train-only candidate vocabulary;
3. replace every occurrence of a selected code word by one opaque token;
4. retain encrypted spaces between words;
5. jointly permute all character and code symbols;
6. canonicalise the resulting surface sequence by first occurrence.

The solver never receives raw pre-permutation labels.

## Corpus and chunking

The existing six hash-pinned Universal Dependencies corpora remain unchanged.

Initial development environment:

- language: English;
- plaintext target: approximately 384 normalized characters;
- chunks end only at word boundaries;
- development: eight non-overlapping corpus `dev` chunks;
- future locked test: twenty untouched corpus `test` chunks;
- candidate nomenclator vocabulary and all language models are derived from corpus `train` only.

## Codebook grid

Development tests:

- codebook sizes: 16, 24 and 32;
- candidate pool: top 96 train words of length at least two;
- fresh codebook per plaintext and key;
- all selected-word occurrences are coded;
- only observed code symbols are scored for mapping recovery.

The primary hard condition is codebook size 24 at approximately 384 plaintext characters.

## Stage A — component oracle gates

### A1. Character-key oracle, codebook unknown

Supplied:

- true character substitution mapping;
- true partition of surface symbols into character and code symbols.

Hidden:

- code-symbol-to-word mapping.

The solver must infer a one-to-one mapping from observed code symbols to the train-only candidate vocabulary using a train-only word n-gram model and consistent recurrence constraints.

Metrics:

- code occurrence word accuracy;
- observed codebook mapping accuracy;
- expanded plaintext character recovery.

### A2. Codebook oracle, character key unknown

Supplied:

- true code-symbol-to-word mapping;
- true character/code symbol partition;
- true set of plaintext character labels represented by observed character symbols.

Hidden:

- assignment of observed character symbols to those plaintext labels.

The solver expands known code words as fixed plaintext spans and solves the residual monoalphabetic key under a train-only character quadgram objective.

Metrics:

- residual character-key accuracy;
- expanded plaintext character recovery.

These two arms isolate codeword inference from character-key search. Neither is a full decipherer.

## Stage B — structure-known joint recovery

Supplied:

- character/code symbol partition;
- true observed plaintext-character inventory, but not its assignment.

Hidden:

- character key;
- codebook mapping.

The solver alternates:

1. residual character-key optimisation under the current codeword assignment;
2. consistent codeword reassignment under the decoded literal-word context;
3. complete plaintext rescoring.

All candidate selection and schedules use development only.

## Stage C — inferred type partition

Stage C is prohibited unless Stage B passes.

The solver must infer which canonical surface symbols are ordinary character symbols and which are nomenclator codes. The known character-alphabet cardinality may be used, but true symbol types may not.

Metrics add:

- code-symbol precision, recall and F1;
- character-symbol precision, recall and F1.

## Language models

- character model: train-only smoothed quadgram plus unigram term;
- word model: train-only interpolated unigram, bigram and trigram probabilities;
- unknown literal test words map to an explicit unknown token;
- candidate code words are restricted to the frozen train-only pool.

No pretrained language model is used in Stage A or B.

## Development gates

Proceed from Stage A to Stage B only if, on the eight English development chunks in the primary condition:

- A1 mean expanded character recovery is at least 90%;
- A1 observed code mapping accuracy is at least 80%;
- A2 mean expanded character recovery is at least 90%;
- neither arm uses the hidden mapping it is intended to recover.

Proceed from Stage B to Stage C only if:

- mean expanded character recovery is at least 70%;
- median recovery is at least 90%;
- at least 7/8 chunks recover at least 70%;
- both character and code mappings improve over their frequency baselines.

## Locked test gate

A development-selected full solver will be evaluated once on twenty untouched English test chunks:

- mean expanded character recovery at least 70%;
- median at least 90%;
- at least 16/20 chunks at least 70%;
- observed code mapping accuracy at least 70%;
- no post-test changes.

## Advancement

Only a locked English pass permits six-language and three-length evaluation. Nulls, spelling variants, multi-token code groups and manuscript-specific rendering remain separate later operators.

Failure at any gate is reported as bounded non-recoverability under the tested assumptions. The gate will not be relaxed and the Voynich Manuscript will not be scored.
