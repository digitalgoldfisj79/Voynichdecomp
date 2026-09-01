# VBM v10 — implementation specification

Date: 2026-09-01
Status: **FROZEN BEFORE FIRST V10 SCORE**
Parent protocol: `VBM_V10_TERMINAL_IDENTIFIABILITY_PROTOCOL.md`

This file resolves operational details left implicit by the already-frozen protocol. It does not change any scientific gate, corpus size, family, language, temperature, chain count, or stopping rule.

## Language model objective

The character 5-gram model scores each decoded line independently as conditional log probability per scored character using 4-character context and additive smoothing `alpha=0.05` over the 26 lowercase ASCII letters.

The optimiser objective is total FIT log likelihood divided by total scored FIT characters. This mean log likelihood is the quantity used in coordinate ascent and simulated-annealing `delta` values.

## Synthetic plaintext

For Stage A each language/replicate first generates one deterministic 2000-line plaintext corpus. Smaller sizes are exact prefixes of that corpus.

Each line contains 8–14 vowel bridges, sampled deterministically. A line is accepted only when all non-empty consonant runs surrounding those vowels occur in the frozen 32-run candidate inventory. Spaces and punctuation are absent before encoding.

## Reveal rule for O1

For each FIT set separately:

- count bridge surface-type occurrences and nucleus surface-type occurrences;
- sort descending by FIT occurrence count, tie-breaking by numeric surface ID;
- reveal `ceil(0.25 * number_of_observed_types)` in each family;
- revealed mappings are fixed to truth and excluded from all optimisation proposals.

No HOLDOUT frequency enters this selection.

## Initialisation

For every chain, each unlocked bridge type is independently drawn from the five vowel values using their frozen LM-bank empirical frequencies. Each unlocked nucleus type is independently drawn from the 32 candidate runs using their frozen LM-bank empirical frequencies. Revealed O1 entries are then overwritten with truth.

Eight chains use deterministic chain-specific seeds.

## Coordinate-ascent pass

A pass visits every observed, unlocked surface type exactly once in a deterministic seed-shuffled order.

For the selected coordinate, all candidate values in its family are evaluated. The value producing the highest global mean FIT log likelihood is retained. Ties within `1e-12` retain the current value.

Two such passes precede annealing; one final pass follows annealing.

## Simulated-annealing sweep

One sweep contains exactly `N_observed_unlocked` proposals, where `N_observed_unlocked` is the number of observed unlocked bridge plus nucleus types.

Each proposal type is selected uniformly from the four protocol-listed classes subject to availability:

1. single bridge reassignment;
2. single nucleus reassignment;
3. bridge-value swap between two unlocked observed bridge types;
4. nucleus-value swap between two unlocked observed nucleus types.

Surface types for proposals are selected with probability proportional to their FIT occurrence counts. Reassignment values are sampled from the frozen family value frequencies, excluding the current value. Swap partners are independently frequency-weighted and distinct from the first type.

The 12 sweep temperatures are geometrically spaced from `0.35` through `0.01`, inclusive.

Acceptance is `min(1, exp(delta/T))` using the change in global mean FIT log likelihood. Downhill moves too extreme for floating-point exponentiation are rejected.

## Chain selection

After the final greedy polish, retain the chain with highest global mean FIT log likelihood. Chain index breaks exact ties.

The true key is never consulted for selection.

## Recovery accounting

Bridge recovery is occurrence-weighted exact value agreement on HOLDOUT events.

Nucleus recovery is occurrence-weighted exact consonant-run-string agreement on non-empty HOLDOUT nucleus events.

Character recovery is factor-event aligned rather than whole-string edit alignment:

- each bridge occurrence contributes one true character and scores 1 if its predicted vowel is exact, otherwise 0;
- each non-empty nucleus occurrence contributes `max(len(true_run), len(predicted_run))` character slots, with position-wise exact matches in aligned prefix positions and unmatched excess positions scored incorrect;
- empty nuclei contribute zero characters.

`REC_CHAR` is total correctly recovered factor characters divided by total factor-character slots. This prevents one wrong variable-length nucleus from shifting all later characters in the line.

`REC_B5` and `REC_N5` restrict to HOLDOUT events whose surface type occurred at least five times in FIT.

`REC_CHAR5` uses the same factor-event character accounting but only for those qualifying surface types; bridges and nuclei are each included only when their own type passes the >=5 FIT-occurrence rule.

## Coverage

`COV_B` and `COV_N` are HOLDOUT occurrence fractions whose surface type appeared at least once in FIT.

## O0

`O0_TRUE_KEY` reports HOLDOUT LM score and recovery sanity checks under the actual generating key. It never enters optimisation or qualification.

## O3

`O3_TRUE_LANGUAGE_FREQUENT_ONLY` is not a separate fit. It is the `REC_*5` view of the O2 fitted map, exactly as implied by the protocol.

## Random dictionary baseline

`RAND_HOLD_LM` is the median HOLDOUT LM score of 20 deterministic random dictionaries drawn from the same language-frequency initialisation distribution used by the solver. Random dictionaries never see HOLDOUT values or truth.

## Split-fit stability

When required, odd-index FIT lines and even-index FIT lines are fitted independently using the same O2 zero-reveal solver and eight-chain procedure. `STAB` is the mean of occurrence-weighted bridge-map agreement and non-empty nucleus-map agreement on HOLDOUT occurrences. Only surface IDs occurring in HOLDOUT contribute; unobserved-in-subfit IDs receive the modal language value before comparison.

## Execution staging

Recovery gates are conjunctive. Therefore Stage-A positive O0/O1/O2/O3 curves are run first. If the O2 recovery gates are already impossible at 2000 lines, adversarial/stability work cannot rescue Stage A and is not required to determine the binding terminal verdict.

If a size satisfies the first four Stage-A recovery criteria, adversarial and stability tests are run at that size before Stage A may qualify. If separation fails, proceed to the next preregistered size.

Stage B remains sealed unless Stage A fully qualifies.
