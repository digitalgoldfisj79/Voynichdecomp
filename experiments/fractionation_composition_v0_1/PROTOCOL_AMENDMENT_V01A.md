# Fractionation Composition v0.1a — development amendment

Status: development-only amendment after the preregistered v0.1 phase-only gate failed. The UD test split and Voynich remain sealed.

## Why v0.1 failed

The v0.1 detector correctly identified all four fractionation arms, but it was not specific. In the runtime-bounded six-language development pilot, 45.83% of negative controls exceeded z >= 3. Expanded monographic substitution exceeded the threshold in 91.67% of samples, expanded+transposed substitution in 62.5%, and the slot control in 25%. Therefore phase periodicity alone is a measurement-degenerate statistic for this question.

This is a detector failure, not evidence for or against Voynich fractionation.

## Amendment rationale

A true coordinate/component fractionator imposes more than periodic phase. The two component streams should be reusable and combine productively across a near-Cartesian pair space. By contrast, verbose monographic expansion may show periodicity while tying both emissions to the same latent plaintext unit.

The amended detector therefore evaluates, for each candidate token/stream mode and block width b=1..8:

1. `phase_MI`: normalized mutual information between surface symbol and the binary component phase;
2. `pair_density`: observed distinct left/right component pairs divided by the Cartesian product of observed left and right component inventories;
3. `pair_NMI`: normalized mutual information between reconstructed left and right component values.

Candidate score:

`phase_MI * pair_density * (1 - pair_NMI)`

The sample statistic is the maximum candidate score over token/stream mode and b=1..8. The full max search is repeated inside every matched-null permutation.

## Control-fairness correction

The v0.1 expanded-monographic controls had larger surface inventories than the coordinate arms. v0.1a removes that mismatch. Slot, Markov, verbose-monographic and verbose-monographic+transposition controls use a surface inventory equal to `rows + columns` for the corresponding plaintext alphabet.

The verbose-monographic control emits two symbols per plaintext character from a small character-specific subset of one shared surface inventory. It has no independent row/column roles. Its transposed version uses a fixed bounded block permutation.

## Matched null

Unchanged from v0.1: exact token lengths and surface symbol counts are preserved, with permutation constrained within L0, L1, R0, R1 and three coarse interior positional strata. The null therefore retains substantial positional grammar while breaking fine component pairing.

## Development gate

Unchanged:

- >=90% positive samples at z >= 3;
- <=10% controls at z >= 3;
- mean positive/control z separation >=2 control SDs.

A pass permits one locked run on the untouched UD test split with the v0.1a detector frozen. A failure stops the branch as non-identifiable. Voynich remains sealed until a locked synthetic test passes.
