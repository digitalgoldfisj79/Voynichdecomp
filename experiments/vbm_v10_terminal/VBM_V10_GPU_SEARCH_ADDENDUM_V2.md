# VBM v10 — GPU exhaustive-block optimiser addendum v2

Date: 2026-09-01
Status: **FROZEN BEFORE ANY V10 SCIENTIFIC OUTPUT**
Parent: `VBM_V10_GPU_SEARCH_ADDENDUM.md`

This amendment is based only on hardware/execution benchmarks. No Stage-A scientific recovery result, synthetic key-recovery statistic, HOLDOUT score, or adversarial result had been exposed when this file was frozen.

It supersedes only the block-size limits in the first GPU addendum. All data generation, oracle conditions, corpus sizes, language material, recovery statistics, chain count, sweep count, partitions, success criteria, adversaries, and stopping rules are unchanged.

## Hardware basis

On HF `a100-large` (NVIDIA A100-SXM4-80GB), the exact Triton implementation of the frozen character-5-gram VBM likelihood scored 2,500,000 complete candidate dictionaries against a 1600-line FIT corpus in 1.1895 s, approximately 2.10 million exact candidates/s, or 1.26 billion exact full-corpus candidates per ten minutes.

The earlier raw integer/tensor enumeration benchmark was explicitly non-scientific and is not used to size production search.

## Final frozen production block sizes

Bridge blocks:
- maximum block size = **8**;
- exhaustive assignment count per full block = `5^8 = 390,625`.

Nucleus blocks:
- maximum block size = **4**;
- exhaustive assignment count per full block = `32^4 = 1,048,576`.

All assignments in every block are evaluated exactly. Outside-block dictionary entries remain fixed.

The three sweep partitions remain exactly as previously frozen:
1. contiguous descending-frequency blocks;
2. frequency-interleaved blocks;
3. deterministic SHA256-seeded shuffled blocks.

Eight deterministic independent chains remain mandatory. Each chain still receives one exact single-site polish before block sweeps, three exact GPU block sweeps, and one final exact single-site polish.

At the full compact Stage-A alphabet (30 bridge types, 96 nucleus types), a nominal three-sweep eight-chain O2 fit therefore evaluates approximately:
- bridge blocks: `ceil(30/8) * 5^8 * 3 * 8 ≈ 37.5 million` exact assignments;
- nucleus blocks: `ceil(96/4) * 32^4 * 3 * 8 ≈ 604 million` exact assignments;
- plus pre/post exact single-site evaluations.

Thus each full Stage-A fit explores roughly **640 million exact conditional key assignments**, without claiming exhaustive coverage of the astronomically larger global key space.

## Why this change is admissible

This is a pre-output compute-strength amendment, analogous to increasing deterministic search depth before data are unsealed. It was chosen from measured exact scorer throughput only, not from recovery performance. It therefore cannot be tuned toward a favourable scientific outcome.

No further block-size, sweep-count, chain-count, partition, or objective change is permitted after this commit.
