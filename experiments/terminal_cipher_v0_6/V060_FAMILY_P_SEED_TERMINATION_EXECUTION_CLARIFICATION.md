# v0.6 Family P — seed-generation termination execution clarification

Date: 2026-07-16

## Defect discovered

The first Family P benchmark shard (`Digitalgoldfish79/6a58d06885d9643ce16d638d`) emitted only its Git commit marker and no scientific row before cancellation.

Audit of `phase_histogram_seeds` found a deterministic non-termination defect for candidate period 2:

- the frozen budget requests eight starting seeds;
- period 2 has only one free shift slot;
- that slot is restricted to its five highest-ranked values;
- therefore at most five distinct initial shift vectors exist;
- the implementation nevertheless required all eight vectors to be unique in an unbounded `while len(seeds) < count` loop.

After the five possible vectors had been collected, termination was impossible. The solver never entered the registered annealing search. The cancelled compute therefore produced no scientific evidence.

## Execution-only correction

The registered budget remains eight starts for every candidate structure.

For each mode and period:

1. Generate distinct histogram seeds until either eight have been obtained or the finite distinct seed space is exhausted.
2. If fewer than eight distinct seeds exist, fill the remaining starts by deterministic seeded sampling with replacement from the same frozen top-five-per-slot proposal distribution.
3. Retain the original `seed_index` in the downstream annealing seed. Repeated initial shift vectors therefore remain independent stochastic starts rather than duplicate trajectories.

No cipher generator, candidate mode, candidate period, scoring function, MDL penalty, iteration count, restart count, threshold, data split, or recovery metric changes.

## Instrumentation

The corrected execution wrapper prints bounded progress markers after each group of eight screening starts and after each refinement candidate. This changes no calculation and prevents future silent non-termination.

This clarification must be committed before rerunning any Family P development shard. The locked test and Voynich data remain sealed.
