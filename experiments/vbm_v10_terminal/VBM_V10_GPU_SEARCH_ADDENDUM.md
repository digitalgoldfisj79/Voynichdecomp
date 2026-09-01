# VBM v10 — GPU exhaustive-block optimiser addendum

Date: 2026-09-01
Status: **FROZEN BEFORE ANY V10 SCIENTIFIC OUTPUT**
Supersedes only the `Stronger global optimiser` implementation paragraph in `VBM_V10_TERMINAL_IDENTIFIABILITY_PROTOCOL.md`.
All data generation, oracle conditions, corpus sizes, recovery statistics, success criteria, adversaries, and stopping rules remain unchanged.

## Rationale

The v9 coordinate solver failed on known-answer synthetic VBM. A terminal identifiability test should not be allowed to fail merely because eight CPU chains explored too little of the discrete key space. GPU compute makes exact enumeration of moderately sized assignment blocks cheap enough to replace single-site annealing with a substantially stronger deterministic block-coordinate search.

This does **not** make the full VBM key space exhaustible. Even Stage-A bridge mappings alone have far more than 10^18 balanced assignments, while 96 nucleus types over 32 values are vastly larger. The GPU is therefore used to exhaustively solve local assignment blocks while maintaining many independent global starts.

## Frozen search

Each O2 fit uses eight deterministic independent chains, as in the parent protocol.

For each chain:

1. frequency-weighted random initial bridge and nucleus assignments;
2. one exact single-site coordinate polish over all observed types;
3. three GPU exact block-coordinate sweeps;
4. one final exact single-site coordinate polish;
5. retain the chain with the highest FIT likelihood.

### GPU block sweeps

A sweep contains both bridge and nucleus blocks.

Bridge blocks:
- observed bridge surface types only;
- ordered by descending FIT occurrence count before each partition is constructed;
- maximum block size = 6;
- every assignment in `{a,e,i,o,u}^b` is evaluated exactly (`<= 5^6 = 15,625` candidates per full block).

Nucleus blocks:
- observed non-empty nucleus surface types only;
- ordered by descending FIT occurrence count before each partition is constructed;
- maximum block size = 3;
- every assignment in the frozen 32-run inventory is evaluated exactly (`<= 32^3 = 32,768` candidates per full block).

The three frozen partitions are:

- Sweep 1: contiguous descending-frequency blocks.
- Sweep 2: frequency-interleaved blocks (rank order `0,k,2k,...,1,k+1,...`, with `k` equal to the number of blocks).
- Sweep 3: deterministic SHA256-seeded shuffle of the observed type list, seed namespace `VBMV10_GPU_BLOCK_SWEEP3::<stage>::<lang>::<rep>::<size>::<chain>::<family>`.

Within each block, all assignments are scored holding every outside-block dictionary entry fixed. The highest-FIT-likelihood assignment replaces the current block. Ties are broken lexicographically by the integer value tuple. No true-key information participates in search or tie-breaking.

Across the three sweeps and eight chains this evaluates millions to tens of millions of exact block assignments per synthetic fit, depending on the number of observed types.

## O1 revealed-key condition

Under O1, revealed entries are immutable and excluded from candidate blocks. Unrevealed entries follow the same block algorithm. O1 remains diagnostic and cannot rescue O2.

## Likelihood implementation

The fixed character 5-gram language model is represented as dense GPU log-probability lookup tensors. Candidate assignments are evaluated in batches. Decoding is performed factor-by-factor so variable-length consonant runs are compacted exactly; padding characters never enter the language-model score.

Only affected FIT lines are rescored for a candidate block. The accepted block score is incorporated into the exact full FIT likelihood cache before the next block.

Batch size may be adjusted automatically to available GPU memory because this changes only execution granularity, not candidates, scores, or decisions. Any out-of-memory retry may reduce batch size only.

## Single-site polish

The pre/post block polish evaluates every possible value for each observed surface type, using the same exact likelihood and affected-line cache. Type order is descending FIT occurrence count. Ties use the same lexicographic rule.

## Chain independence

Initial maps are deterministic from SHA256 namespaces and do not depend on previous scientific results. A chain cannot seed another chain. The best chain is selected solely by FIT likelihood.

## Cumulative corpus sizes

The same 2000-line positive replicate is generated once and prefixes are used at 100, 250, 500, 1000, 2000 lines exactly as frozen. Fits at a larger size are **not warm-started** from a smaller-size recovered map; each size receives eight fresh deterministic chains. This prevents the phase-transition curve from inheriting information from prior sizes.

## Output firewall

The first scientific V10 output may be emitted only after this addendum and the executable implementing it are committed. Hardware-only kernel benchmarks on random tensors are permitted beforehand and have no scientific status.
