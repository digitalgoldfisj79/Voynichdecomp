# Recoverability frontier v0.5.5 — protocol amendment A: minimal equivalent period

Date: 2026-07-16

Status: fixed after the component smoke exposed exact representational ties, before Stage B construction.

## Structural equivalence

A repeated block permutation can have more than one exact block-size representation. In particular, any size-four permutation applied independently to consecutive four-symbol blocks has an equivalent size-eight representation formed by repeating the same permutation in each half of the eight-symbol block.

The resulting plaintext and language-model score are identical. No algorithm can distinguish these parameterisations from ciphertext alone.

## Canonical convention

The transposition structure is defined by its **smallest equivalent block period** within the candidate set.

- Candidate block sizes remain ordered `4, 6, 8`.
- Candidates within numerical score tolerance `1e-9` are treated as an equivalence class.
- The selected block size is the smallest size in the best-scoring equivalence class.
- Exact permutation accuracy is evaluated against the canonical minimal-period representation.
- Score margin is measured against the best non-equivalent plaintext candidate, not merely the next parameter tuple.

The current implementation already enumerates sizes in ascending order and uses stable sorting, so its selected size follows the minimal-period convention. Future reports must not interpret a zero raw second-candidate margin as evidential uncertainty when the tied candidate yields the same detransposed plaintext.

## Generator restriction

For block sizes six and eight, generated permutations must not themselves reduce to a smaller candidate period. If such a reducible permutation is sampled, it is rejected and resampled. This restriction affects future trials only; the current development trials are audited for reducibility before Stage B.
