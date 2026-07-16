# Recoverability frontier v0.5.5 — Stage A and initial Stage B result

Date: 2026-07-16

Verdict: **BOTH COMPONENT GATES PASS; INITIAL JOINT COORDINATE SEARCH FAILS DEVELOPMENT. ONE STRATIFIED SEED EXPANSION IS PERMITTED.**

No Voynich text was scored.

## Corrected family

The synthetic family combines:

- a fresh monoalphabetic substitution key;
- a fresh non-identity permutation repeated within blocks of size 4, 6 or 8;
- a fresh joint relabelling of the substitution alphabet;
- first-occurrence recurrence canonicalisation;
- no padding, block markers or exposed word boundaries.

The scored plaintext length is exactly 384 characters, divisible by every candidate block size.

## Stage A1 — true substitution key; hidden block size and permutation

Job: `Digitalgoldfish79/6a5869afb1669a49bf076cc5`

The complete 41,064-candidate space was enumerated for each trial using the train-only character quadgram objective.

Across 24 English development trials:

- mean character recovery: **100%**;
- block-size accuracy: **24/24**;
- canonical exact permutation accuracy: **24/24**;
- true candidate rank: **1 in every trial**;
- mean enumeration time: approximately 1.5 seconds per ciphertext.

A size-four repeated permutation has an exactly equivalent size-eight representation. The protocol therefore defines the canonical structure as the smallest equivalent block period. Stable ascending enumeration already implements this convention.

**A1 verdict: pass.**

## Stage A2 — true transposition; hidden substitution key

The first A2 output from job `6a5869afb1669a49bf076cc5` is excluded because the mono solver was mistakenly bound to the homophonic quadgram objective. That implementation defect was recorded before rerunning A2.

Valid replacement job: `Digitalgoldfish79/6a586ac0b1669a49bf076ce5`

Scientific SHA-256: `e2c96591ab4d122c75640015e83c0d966003a6e0b6d0d1e6838d34347b4578cc`

The rerun used the unmodified passing v0.5.1 monoalphabetic solver: train-only trigram plus unigram objective, `700,000 × 50`.

Across 24 trials:

- mean character recovery: **99.7070%**;
- median: **100%**;
- minimum: **98.1771%**;
- all 24 trials exceeded 98%;
- mean key accuracy: **97.6496%**;
- exact plaintext recovery: 18/24.

Every block-size-specific A2 gate passed.

**A2 verdict: pass.**

## Cheap Stage B screen

Job: `Digitalgoldfish79/6a586b91b1669a49bf076cf9`

Scientific SHA-256: `8e2bf77aee8648cc7ffa14c9aec22d30e6786da9118714dfa76ba361c027dbee`

A frequency-derived key ranked all transposition candidates:

- median true rank: 11;
- mean true rank: 825.6;
- maximum: 9,857;
- top-64 recall: 58.3%;
- top-256 recall: 70.8%;
- top-1,024 recall: 75.0%.

The frequency screen was therefore retained only for seeds and never used as a hard prune.

## Initial Stage B coordinate solver

Job: `Digitalgoldfish79/6a586cd885d9643ce16d5d94`

Scientific SHA-256: `109a088738be8eb9996dd116f84208d4371e527743dec597be6bbe8dfa0e7750`

Frozen schedule:

- 16 globally top-ranked frequency seeds;
- 16 deterministic random full-space seeds;
- two coordinate cycles;
- each cycle: `50,000 × 5` mono search followed by complete transposition enumeration;
- final `700,000 × 50` mono refinement and one complete re-enumeration.

Results across 24 development trials:

- mean recovery: **57.9753%**;
- median: **32.5521%**;
- trials at least 70%: **11/24**;
- trials at least 90%: **11/24**;
- exact plaintext recovery: 3/24;
- canonical block-size accuracy: **75.0%**;
- exact permutation accuracy: **45.8333%**.

By block size:

| Block size | Mean recovery | Trials ≥70% | Block-size accuracy | Exact permutation |
|---:|---:|---:|---:|---:|
| 4 | 50.9766% | 3/8 | 37.5% | 37.5% |
| 6 | 62.8906% | 4/8 | 87.5% | 50.0% |
| 8 | 60.0586% | 4/8 | 100% | 50.0% |

When the correct joint basin was reached, recovery was normally 99.5–100%. The global top-seed list was dominated by size-eight candidates, materially disadvantaging the canonical size-four family. Several failures also occurred despite relatively high frequency-screen ranks, so rank alone is not an adequate candidate selector.

**Initial Stage B verdict: fail.**

## Permitted final development escalation

The frozen protocol permits varying seed count and short-cycle budget on development data. The next and final coordinate-search configuration therefore changes only those dimensions:

- stratify seeds by block size;
- 16 top frequency-screened and 16 deterministic random seeds **within each of sizes 4, 6 and 8** = 96 starts;
- three coordinate cycles;
- `100,000 × 8` mono search per cycle;
- unchanged complete transposition enumeration;
- unchanged final `700,000 × 50` refinement;
- unchanged development gates.

No objective, corpus, ciphertext, true-key access or gate is changed. Failure of this expanded schedule closes the v0.5.5 joint solver without a locked test.