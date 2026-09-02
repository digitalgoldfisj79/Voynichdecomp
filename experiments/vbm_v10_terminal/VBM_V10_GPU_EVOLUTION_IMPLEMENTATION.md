# VBM v10 — GPU evolutionary implementation freeze

Date: 2026-09-02
Branch: `experiment/vbm-v10-terminal-identifiability-20260901`
Status: **FROZEN BEFORE BINDING STAGE-A OUTPUT**

This note records execution details for the already-frozen `VBM_V10_GPU_EVOLUTION_ADDENDUM.md`. It does not alter the scientific protocol, synthetic corpora, oracle ladder, candidate domains, recovery statistics, success thresholds, or stopping rules.

## Executable

The production source is `vbm_v10_stage_a_gpu_evolution.py`, embedded losslessly in `vbm_v10_stage_a_gpu_evolution_bootstrap.py` for Hugging Face execution.

Decompressed production-source SHA256:

`d64bb3a63f0c17cfc9326ca45336d12ea462feb15544a7094398fb022445e95c`

The bootstrap verifies this hash before executing the embedded source.

## Deterministic candidate generation

All random-looking choices are deterministic functions of SHA256-derived 31-bit namespace seeds and candidate IDs. This is an implementation detail only; no true-key information enters candidate generation.

- Initial complete keys: deterministic hash of candidate ID × coordinate.
- Parent selection: fixed integer CDF for weights proportional to `1/sqrt(rank+1)`.
- Mutation coordinates: choose the required number of active coordinates by deterministic hash rank, giving distinct coordinates without replacement.
- Mutation values: deterministic hash into the legal alternative domain, explicitly excluding the current value.
- Fresh 5%: an affine permutation of the 2,500,000 candidate IDs modulo the population size marks exactly 125,000 IDs as fresh each generation. Those rows are generated as complete independent keys, not mutated offspring.
- O1 revealed coordinates are overwritten with truth immediately and are absent from the mutation-active set.

Candidate generation is invariant to GPU batch partitioning; this is checked in the pre-binding self-test.

## Exact scoring and selection

Every candidate used for selection is scored by the exact frozen character-5-gram FIT objective in Triton, with variable-length nucleus strings, bridge vowels, and true line resets.

Population scoring is batched only for memory control. Per-batch candidates are reduced to exact top candidates, and the global elite reduction re-applies the same rule. Exact score ties are broken by the full 126-entry key in ascending lexicographic order, as frozen in the addendum.

## Search budget

Per independent chain:

- 2,500,000 complete keys in generation 0;
- 60 × 2,500,000 complete keys in evolution generations;
- total population evaluations = 152,500,000 complete keys;
- then exact one-entry coordinate polish, capped at 40 accepted changes.

There are eight independent chains. Hardware parallelism changes wall time only.

## Pre-binding qualification

Before any binding Stage-A row is emitted, the executable must pass:

1. exact GPU versus frozen CPU score agreement on fixed synthetic keys, tolerance `1e-5`;
2. deterministic candidate generation invariance to batch partitioning.

The earlier frozen smoke qualification already established exact GPU/CPU agreement and monotone coordinate-polish likelihood improvement on a separate synthetic smoke corpus. This production self-test is an additional execution check, not a scientific gate.

## Output recovery

Each completed Stage-A row is emitted both as JSON and as a zlib-compressed protocol-5 pickle encoded in base64. The final row set is emitted the same way. This makes each result atomic and recoverable from job logs.

No Voynich plaintext, TRAIN, H1, or C1 data are accessed by this executable.
