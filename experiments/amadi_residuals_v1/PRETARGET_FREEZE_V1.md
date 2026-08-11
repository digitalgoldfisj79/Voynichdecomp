# Amadi Residuals v1 — Pretarget Freeze

Date: 2026-08-11
Status: **BINDING FREEZE BEFORE FORMAL Q1 AND BEFORE ANY H2 CIPHER SCORE**

## Frozen executable

Code commit at final executable freeze: `dfb533899c0bdcd9ae739c933a61de55c0d314d7`

Entry point:
`experiments/amadi_residuals_v1/amadi_driver_release_v1.py`

The release entry point imports the following immutable execution chain:

| file | SHA-256 |
|---|---|
| `amadi_residuals_v1.py` | `0b3bdfd9168da5e0aa270cff0050c786c8563583a169f3eb3e6e2065f96cb798` |
| `amadi_driver_v1b.py` | `255bdb2afc372216f2becef80fc54aa22b1526414b2988d6fd994011b21df79d` |
| `amadi_driver_v1c.py` | `0f39442a0141ed2e5ab2e40a24e2c2fa2f7aafbc1fc64d073d2660b4d05be2a3` |
| `amadi_driver_final_v1.py` | `eade09e0c92a60abec7cddf8ef52475f65f6b75f3f266f73b7361cc122a547be` |
| `amadi_driver_release_v1.py` | `ad4ee1c9c7aac0ef6b14309cc16b85588b7a8fb1c5d7d95e0836a429a32951a7` |

Concatenated executable-bundle SHA-256:
`a51fe37a1e00a7f4c6189e481d077f0fda62f48d6dd4ff8a5216e1c8f11fe2f4`

The final release wrapper only restores the exact HTTP request headers already used successfully by Cipher Coverage v1. It changes no parser, representation, scorer, solver, threshold or split.

## Frozen source and target representation

RF source SHA-256:
`eb857a1f353b18983fbc25b954e1bbce227a26d99cefabfda9206ff9b57644d2`

Preflight census:

- pages: 227
- raw words: 37,848
- retained words: 37,647
- total alphabetic positions: 194,617
- retained positions: 193,776
- coverage: 0.9956786919950467
- uncertain words excluded: 88
- rare-symbol words excluded: 113

Split is frozen in `TARGET_SPLIT_MANIFEST_V1.json`:

- FIT-A = old T1 + old H1: 181 folios / 161,296 chars
- H2 = untouched half of old C1: 23 folios / 15,224 chars
- C2 = remaining untouched half: 23 folios / 17,256 chars; **SEALED**

## Frozen admitted family inventory

Primary families entering formal qualification:

1. `R12H / R12_V1_024` source-derived deterministic formalisation, Italian only; global surjective 19→12 observable-to-latent assignment.
2. `VC_END` exact prose operation; transformed-language scorer.
3. `PWA_K`, K = 2,3,4,5; one global bijection per modulo-word-position state, phase resets at every word.
4. `GHOUSE5`; five document-global substitution maps keyed by target selector class `{k,t,p,f,NONE}`.

Stopped before Q1:

- R12_V3_390: source underdetermined
- R12_V4_454: source underdetermined at G/P branch
- plaintext autokey: source underdetermined
- walking/two-stream direct RF: source underdetermined/carrier-dependent
- exact NTRC: surface incompatible at 0.995 gate
- exact modulo-105: surface incompatible at 0.995 gate
- Glorioso: not a direct openly segmented RF family
- syllable mutation: no unique target schedule
- dual-meaning: non-identifiable without external constraint

No family may be added after this freeze.

## Frozen source-fidelity reconciliation

`SOURCE_Q0_RECONCILIATION_V1.md` and `SOURCE_RECONSTRUCTION_CORRECTION_V1.md` are binding.

- VC_END: the literal prose operation matches 28/34 tabulated word pairs; the six predeclared source discrepancies are retained as discrepancies and are not used to alter the algorithm.
- R12 section 024: short examples are rule-local; computational transform is a deterministic composition of the explicit operations.
- `g` is deleted, not `g->i`, as shown consistently by the worked examples.

Any new source discrepancy discovered after this freeze blocks the affected arm; it does not permit further rule editing.

## Frozen statistical model

Primary scoring model: word-sensitive order-3 character LM with explicit boundary padding and add-0.25 smoothing.

Training/control sentence residues remain disjoint:

- LM train: sentence-index residues `{0,1,3,4,6,8}` mod 10
- controls: `{2,5,7,9}` mod 10

Languages: Latin, Italian, German, French, Ancient Greek, Hebrew, Arabic, Spanish. R12H is Italian-only in v1.

## Frozen optimizer

For formal qualification and target fitting:

- 30,000 legal proposals per restart
- maximum 12 restarts per ensemble
- restart batches of 4
- two independent ensembles
- convergence: objective difference <= 1e-7 nats/transformed character AND occurrence-weighted assignment agreement >=0.95
- no optimizer extension after Q1 begins

The prefreeze full-budget development check recovered fresh controls as follows:

- PWA K=2: 1.0000
- PWA K=3: 0.99335
- PWA K=4: 1.0000
- PWA K=5: 0.99667
- GHOUSE Latin: 1.0000
- GHOUSE Italian: 0.97333

All six converged in the first 4-restart batch with A/B agreement 1.0. These are development controls only, not formal Q1 observations.

## Frozen formal gates

### Q1 recovery
Per exact rule/state count >=3 controls; >=1,200 transformed units fit + >=1,200 holdout.

Each rule:
- all controls converge
- median recovery >=0.95
- minimum recovery >=0.85
- median A/B agreement >=0.95
- minimum agreement >=0.90

### Q2 blind recognition
For admitted non-R12 families:
- family accuracy >=0.90
- PWA exact-rule accuracy >=0.85 **when PWA is in the admitted Q2 universe; otherwise N/A, not zero**
- language accuracy >=0.90
- median recovery >=0.90
- any language with >=4 controls must achieve >=0.75 language accuracy

R12H is separately judged on Italian recovery/convergence, per the parent protocol.

### Q3 absolute calibration
For each qualified family×language cell:
- 8 fresh controls
- `ABS_FLOOR` = 5th percentile held-out true-family score
- `DELTA_FLOOR` = 5th percentile held-out family score minus matched M0 baseline
- for PWA, also freeze `RESET_DELTA_FLOOR` = 5th percentile true word-reset score minus deterministic phase-shuffled score

### Q4 structured negatives
80 total, 16 each:
- iid unigram
- order-2 Markov
- motif repeat/mutate
- copy/mutate
- slot grammar

Pass only if:
- <=2/80 false positives total
- <=1 false positive in any generator class

## Frozen family-specific H2 gates

### R12H
Positive interpretation additionally requires assignment agreement >=0.95.

### PWA
Must reach ABS and DELTA floors and beat phase-shuffled control by the frozen `RESET_DELTA_FLOOR`.

### GHOUSE5
Target selector extraction:
- leftmost `k/t/p/f` in each retained word is selector
- remove exactly that first occurrence from payload
- later gallows remain payload
- no marker -> `NONE`

For a positive interpretation:
- all five selector states must have >=500 FIT-A payload characters
- per-state A/B map agreement >=0.90 for every state
- H2 real selector assignment must score strictly above the 99th percentile of 256 deterministic **within-H2-folio** selector-label permutations preserving each folio's selector-class counts
- ABS and DELTA floors and Q4 must also pass

These selector requirements are positive-specific. A converged family winner below its ABS floor may still support a formal negative under the parent protocol.

## One-way rule

After this freeze:

- no code changes
- no threshold changes
- no language changes
- no family/rule deletion to improve performance
- no representation substitution
- no optimizer increase
- no H2 plaintext inspection
- no C2 access unless H2 produces a fully admissible candidate

If an implementation failure occurs after Q1 begins, the affected arm is `IMPLEMENTATION_BLOCKED` unless the correction is purely infrastructural and provably leaves the scientific executable bundle unchanged. Scientific code changes require a new version, not a continuation of v1.