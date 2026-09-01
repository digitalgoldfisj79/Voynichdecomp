# VBM Joachim-exact v9 — Q1 synthetic known-answer key-transfer calibration

Date: 2026-09-01
Status: **FROZEN BEFORE Q1 SYNTHETIC OUTCOMES; H1/C1 REMAIN UNREAD**

Q0 and Q0b passed. Q1 asks whether a solver can recover and transfer the newly clarified VBM whole-nucleus codebook when a reusable key is known to exist.

Q1 contains **no Voynich plaintext/language fit**. Voynich is used only to freeze the empirical surface-type inventory/frequency profile from the already-open Q0 TRAIN pool. H1 and C1 remain excluded and unread.

## Why Q1 is mandatory

The source-faithful parser has a large mapping space: thousands of nucleus surface types, variable 1–5-consonant outputs, and homophonic vowel bridges. A model this expressive is not evidentially meaningful merely because a fitted sentence can be produced. Before target fitting, the inference instrument must show that a global reusable key can actually be learned from ciphertext and transferred unchanged to unseen ciphertext.

## Plaintext corpora

Two independently normalized language corpora are used so the solver cannot qualify only because it was engineered around German:

- German: Universal Dependencies `UD_German-GSD`, train CONLLU.
- Italian: Universal Dependencies `UD_Italian-ISDT`, train CONLLU.

ASCII normalization is frozen:
- Unicode transliteration to ASCII;
- lowercase;
- `j -> i`, `v -> u`, `w -> u`, `y -> i`, `x -> s`, `z -> s`;
- retain `a-z` only.

Vowels are exactly `{a,e,i,o,u}`. A plaintext stream is represented as an alternating event stream of individual vowel events and maximal non-empty consonant runs. Consonant runs longer than five are ineligible and break the stream; they are not split or repaired.

Sentence-index residue modulo 10 is frozen:
- 0–5: language-event LM training only;
- 6: DEV engineering plaintext;
- 7: CAL qualification plaintext;
- 8: VAL one-shot confirmation plaintext;
- 9: unused reserve.

CAL and VAL are never used to tune solver parameters.

## Semantic vocabulary and LM

For each language independently:
- all five vowels are semantic bridge states;
- candidate nucleus states are the **64 most frequent non-empty consonant runs of length 1–5** in that language's LM-training split, ties lexicographic;
- plaintext segments used in DEV/CAL/VAL must be representable entirely by those 64 nucleus states plus the five vowels; non-representable segments are excluded before encryption.

The semantic language model is a boundary-aware add-0.25 smoothed trigram model over these 69 event states plus a line boundary state. No surface labels enter LM construction.

## Empirical VBM surface inventory

Use the source-corrected Q0b parser and only the original Q0 TRAIN folios.

Active surfaces:
- nucleus surface types with TRAIN frequency >= 5;
- bridge surface types with TRAIN frequency >= 3.

Their empirical Q0 TRAIN counts are retained as sampling weights. H1, C1, and Q0 INTERNAL_HOLDOUT are not used to construct the surface inventory.

## Synthetic key construction

For each language and replicate, construct a deterministic seeded global homophonic key:
- every one of the 64 nucleus semantic states is assigned at least one active nucleus surface type;
- every one of the five vowel states is assigned at least one active bridge surface type;
- remaining surface types are assigned to semantic states by seeded draws proportional to LM-training semantic frequency;
- when encrypting a semantic event, choose among surface types assigned to it with probability proportional to their frozen Q0 empirical count.

This produces a many-surface-to-one-semantic key matching the VBM hypothesis class.

Each replicate contains:
- FIT: 6,000 semantic events;
- HOLDOUT: 3,000 semantic events;
with line boundaries retained and no plaintext overlap between the two windows within a replicate.

## Positive and hostile negative families

For both German and Italian:

**GLOBAL**
- one synthetic key is used for FIT and HOLDOUT.

**FRESH**
- FIT is identical to the corresponding GLOBAL FIT ciphertext/plaintext pair;
- HOLDOUT plaintext is identical to GLOBAL HOLDOUT;
- HOLDOUT is re-encrypted under an independently seeded key drawn from the same key generator.

Thus GLOBAL and FRESH have matched plaintext, parser, vocabulary, homophony, event counts, and surface-generation mechanism. They differ only in whether the FIT key transfers.

## Replicate namespaces

- DEV: replicates 0–3 per language; may be inspected for solver engineering only.
- CAL: replicates 100–105 per language; first inferential qualification set.
- VAL: replicates 200–205 per language; one-shot confirmation set.

No seed may be replaced because of an inconvenient result.

## Solver class

The solver receives only:
- FIT ciphertext event sequences;
- event class of each surface (nucleus vs bridge);
- frozen language-event trigram LM;
- frozen candidate semantic states;
- frozen codebook cost;
- Q0 empirical surface frequencies for initialization only.

It does **not** receive the synthetic truth key or plaintext during fitting.

Mapping constraints:
- bridge surfaces may map only to the five vowels;
- nucleus surfaces may map only to the 64 frozen consonant-run states;
- mapping is deterministic surface -> semantic state;
- homophony is unrestricted;
- no context-, line-, folio-, position-, or length-specific alternate key.

Objective:

`FIT log P_LM(decoded events) - key_cost_bits * ln(2)`

with frozen key cost:
- bridge mapping: `log2(5)` bits per mapped surface type;
- nucleus value of consonant length `l`: `log2(5) + l*log2(21)` bits per mapped surface type.

The objective may be optimized by deterministic/reproducible coordinate ascent, annealing, or equivalent engineering. **Only DEV may be used to improve optimizer implementation.** Scientific model class, candidate vocabulary, corpus splits, key generator, costs, and gates cannot change.

Before CAL is run, the exact solver code commit is frozen. No tuning is allowed after any CAL outcome is visible.

## Metrics

For each replicate and family:

1. FIT event-weighted mapping accuracy against hidden truth (audit only; truth revealed only after fitting).
2. HOLDOUT semantic-event accuracy from the unchanged FIT mapping.
3. HOLDOUT LM NLL in nats/event under inferred mapping.
4. HOLDOUT true-key LM NLL ceiling.
5. inferred-key regret = inferred NLL - true-key NLL.
6. paired GLOBAL minus FRESH HOLDOUT semantic accuracy.
7. paired FRESH minus GLOBAL HOLDOUT NLL (positive favours reusable-key discrimination).
8. fraction of HOLDOUT events whose surface type occurred in FIT.

Unknown HOLDOUT surface types receive no fitted mapping and are excluded from semantic-accuracy/NLL numerators and denominators after resetting LM context. Their fraction is reported. No mapping may be inferred from HOLDOUT.

## DEV requirement before freezing solver

DEV is engineering-only. The solver may be improved until, simultaneously in both German and Italian:
- median GLOBAL HOLDOUT semantic accuracy >= 0.50;
- median paired GLOBAL-FRESH accuracy gap >= 0.20.

If this cannot be achieved without changing the scientific model class, Q1 closes as **INSTRUMENT_NOT_QUALIFIED** and H1/C1 remain sealed.

## CAL binding gates

Using six CAL replicates per language, all of the following must hold separately for German and Italian:

1. median GLOBAL HOLDOUT semantic accuracy >= **0.60**;
2. every GLOBAL replicate HOLDOUT semantic accuracy >= **0.45**;
3. median FRESH HOLDOUT semantic accuracy <= **0.30**;
4. median paired GLOBAL-FRESH semantic-accuracy gap >= **0.30**;
5. at least 5/6 paired GLOBAL-FRESH accuracy gaps are > **0.20**;
6. median GLOBAL inferred-key regret <= **0.35 nats/event**;
7. median paired `(FRESH NLL - GLOBAL NLL)` >= **0.25 nats/event**;
8. median HOLDOUT scored-event fraction >= **0.90**.

Failure of any CAL gate closes Q1. VAL stays sealed and H1/C1 remain sealed.

## VAL one-shot confirmation gates

If and only if CAL passes, run six fresh VAL replicates per language with the frozen CAL solver commit.

All CAL thresholds above apply unchanged to VAL. No retuning, restart-budget increase, seed replacement, or candidate-vocabulary change is allowed after CAL.

Q1 passes only if **both languages pass both CAL and VAL**.

## What Q1 can and cannot establish

Q1 PASS establishes only that the v9 inference machinery can learn and transfer a global reusable whole-nucleus/homophonic VBM key under matched synthetic conditions and can discriminate it from a fresh-key adversary.

It does not establish that Voynich is a cipher, that VBM is correct, that the language is German/Bavarian, or that Joachim's supplied feasibility plaintext is genuine.

Only after Q1 PASS may an H1 language/key-transfer protocol be preregistered. C1 remains sealed until H1 independently passes its own gates.

## Stop rules

- No H1 or C1 access in Q1.
- No threshold relaxation.
- No CAL/VAL seed replacement.
- No language-specific rescue parser.
- No contextual key added after failure.
- No use of synthetic truth during optimization.
- DEV engineering changes must be committed before CAL.
- A CAL failure is binding and prevents VAL/target access.
