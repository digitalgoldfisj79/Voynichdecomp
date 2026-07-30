# Result schema

## `directional_observations.csv`

One row per candidate-reference × probe × compressor × representation.

Core fields:

- target corpus/document/probe identity;
- candidate corpus;
- candidate and own-source reference documents;
- representation and compressor;
- candidate conditional bits per probe byte;
- own-source conditional bits per probe byte;
- directional excess bits per probe byte;
- candidate rank, own-source rank and predicted corpus;
- probe, reference and compressed sizes;
- probe SHA-256.

## `ncd_pairs.csv`

One row per unordered corpus-reference pair × compressor × representation, retaining:

- forward NCD;
- reverse NCD;
- registered symmetric mean;
- concatenation-order gap.

## `summary.json`

Contains only deterministic scientific metadata and aggregate cells. Runtime metadata is excluded from the scientific-payload hash.

## `validation.json`

Independent recomputation of directional excess, ranks, winners and output hashes.

## `consensus.json`

One decision per independent probe, including all compressor/representation votes, abstentions, coverage, accuracy and target-wise recall.

## Trees

UPGMA trees are deterministic presentation derivatives of a specified NCD matrix. The exact compressor and representation are encoded in the filename. A tree is never treated as an independent statistical test.
