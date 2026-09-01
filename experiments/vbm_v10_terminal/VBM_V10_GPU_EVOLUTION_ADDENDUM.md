# VBM v10 — exact-likelihood GPU evolutionary optimiser addendum

Date: 2026-09-01
Status: **FROZEN BEFORE ANY BINDING V10 SCIENTIFIC OUTPUT**

This addendum supersedes only the optimiser implementation in `VBM_V10_GPU_SEARCH_ADDENDUM.md`. The parent V10 architecture, synthetic data, corpus sizes, oracle ladder, recovery statistics, adversarial families, success thresholds, and stopping rules remain unchanged.

## Motivation

HF benchmark on one A100-SXM4-80GB established that the exact V10 character-5-gram likelihood can score approximately 2.10 million complete candidate keys per second on the full 1,600-line FIT portion of a 2,000-line Stage-A corpus. A search instrument should therefore spend compute on large exact-likelihood populations rather than small CPU-style local searches.

## Independent chains

The frozen eight-chain requirement is retained. Each chain has an independent deterministic SHA256 seed namespace and cannot inherit a key, elite set, score, or mutation state from another chain. When multi-GPU hardware is available, chains may execute concurrently on separate GPUs. Hardware parallelism does not change the search path of any chain.

## Candidate representation

Each candidate consists of one complete global VBM dictionary:

- 30 bridge entries, each in `{a,e,i,o,u}`;
- 96 nucleus entries, each in the frozen 32-run language inventory.

No context-specific or line-specific assignments are permitted. Solver candidates are not constrained to the balanced homophone multiplicities used by the synthetic generator; this is deliberately favourable to the solver and matches v9/v10's unrestricted inverse problem.

## Exact scoring

Every candidate used for selection is scored with the exact frozen character-5-gram FIT likelihood, including:

- variable-length nucleus strings;
- bridge vowels;
- true line resets;
- no cross-line n-grams;
- average log likelihood per scored character, matching the frozen V10 objective.

No surrogate score is allowed to select an elite. Execution may batch candidates and lines but may not approximate the likelihood.

## Population budget per chain

Population size per generation: **2,500,000 complete keys**.

Elite set retained after each generation: **4,096 keys**.

Initial generation:

- 2,500,000 deterministic full random keys;
- values generated independently within the correct bridge/nucleus domains;
- select top 4,096 by exact FIT likelihood.

Evolution generations: **60**.

Thus each chain evaluates 152.5 million complete keys before the final coordinate polish. This budget is fixed independent of intermediate recovery or likelihood.

## Offspring generation

For every non-initial generation, candidate IDs deterministically select a parent from the previous 4,096 elites. The parent-selection distribution is rank-weighted with weight proportional to `1/sqrt(rank+1)`.

Mutation count is frozen by generation:

- generations 1–10: 12 dictionary entries;
- generations 11–25: 6 entries;
- generations 26–45: 3 entries;
- generations 46–60: 1 entry.

Mutation positions are sampled without replacement from the 126 dictionary entries. Each selected entry receives a uniformly selected alternative value from its own family, excluding its current value.

To preserve global exploration, exactly 5% of every generation (by deterministic candidate-ID hash) are fresh full random keys rather than mutated elites.

No crossover is used. This keeps the search operator interpretable and avoids adding another free design choice.

## Elitism and ties

The previous generation's 4,096 elites are inserted unchanged into the next generation before truncation. The next elite set is the top 4,096 exact FIT scores among elites plus offspring.

Score ties are broken by lexicographic order of the 126 integer mapping values. No truth-key information enters tie-breaking.

## Final exact coordinate polish

After generation 60, the single best key in each chain receives exact greedy single-entry coordinate polish.

At a coordinate step, every legal one-entry alternative is evaluated:

- each bridge entry: 4 alternatives;
- each nucleus entry: 31 alternatives.

The highest exact FIT-likelihood improving neighbour is accepted. Repeat until no improving neighbour remains or 40 accepted coordinate changes have occurred. The 40-change cap is execution-bounding and is frozen before binding output.

The chain result is its post-polish key.

## Oracle O1

When O1 reveals true mappings, revealed entries are immutable:

- they are fixed in all initial random keys;
- they cannot be mutation positions;
- they are excluded from coordinate polish.

O1 remains diagnostic and cannot rescue O2.

## Corpus-size independence

Each Stage-A size (`100,250,500,1000,2000`) starts from eight fresh deterministic chain populations. No elite or recovered map transfers between sizes.

## Smoke qualification

Before opening any binding Stage-A replicate, the executable must pass two non-binding implementation checks:

1. exact GPU scorer agrees with the original CPU V10 scorer on the same fixed keys to absolute score tolerance `1e-5`;
2. on a separate `SMOKE` synthetic global-VBM corpus not used in Stage A, a deliberately truth-seeded key with 10 randomly corrupted dictionary entries is improved by the final coordinate-polish routine and never decreases FIT likelihood.

These are software/instrument checks only and do not qualify the scientific method.

## Binding Stage-A execution order

After smoke qualification, run all six Stage-A positive replicates at all five frozen corpus sizes under O2. O1/O3 and adversarial controls are computed as required by the parent protocol. Execution order may be parallel, but no result changes the fixed search budget.

## Interpretation

Failure after this search budget is not a proof that no conceivable optimiser could invert VBM. It is the frozen practical stopping test: if 152.5 million exact complete-key evaluations per independent chain plus exact coordinate polish cannot meet the synthetic recovery gates at or below Voynich-scale data, the present VBM receives no Voynich decipherment run.
