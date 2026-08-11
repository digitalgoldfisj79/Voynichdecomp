# VBM v6 — topology-preserving cipher fingerprint / cross-folio key transfer

Date: 2026-08-11
Branch: `experiment/vbm-key-transfer-v6-20260811`
Status: PREREGISTERED BEFORE v6 TARGET SCORING

## Scientific question

Does the frozen VBM surface representation contain reusable cross-folio symbol-identity information consistent with a stable cipher transducer, over and above its already-established Bavarian-like C/V topology?

v5 established two facts that are both binding background:
1. the VBM C/V topology is strongly Bavarian-directional under an independently transferred classifier;
2. the full 144-symbol stream fails a Bavarian/German homophonic-language-vs-surface-null test on H1.

v6 therefore does **not** ask whether the whole stream is explained by the v5 HMM. It asks whether a reusable key-like mapping survives after conditioning on topology.

## Frozen target disposition

- `H1 = f28v f31v f88r f5r f34r f81v` is already consumed and is not used as a v6 confirmation set.
- `C1 = f85r1 f53v f33r f10r f23r f111r` remains sealed.
- v6 development/target-discovery uses the existing 181-folio VBM FIT pool only after synthetic qualification.
- C1 may be opened exactly once only if every v6 qualification and FIT gate below passes.
- No target plaintext or decoded string may be inspected at any stage.

## Frozen representation

Use the existing VBM geometry without alteration:
- 21 core surface units;
- 123 bridge surface units;
- core/bridge typing fixed by the v1/v5 representation;
- RF source and 0.995 bridge occurrence coverage unchanged.

All topology-preserving permutations independently permute labels `0..20` and `21..143` within each folio while leaving event order, C/V type sequence, line/segment boundaries, event counts, and within-folio repetition pattern unchanged.

## Instrument

For each train/held-out folio split and each candidate language `L in {Bavarian, German}`:

1. fit the existing moment-initialised typed soft-HMM emission matrix `E_L` on train folios only;
2. select the candidate language using **train likelihood only**;
3. freeze `L*` and `E_L*`;
4. score untouched held-out folios under the frozen map;
5. rescore 24 independent topology-preserving per-folio label permutations with the same frozen map.

Primary statistic per split:

`ITE = score_observed_holdout - median(score_topology_preserved_permutations)`

This is the identity-transfer excess: predictive value carried by stable surface-symbol identity beyond the complete C/V topology and within-folio repetition pattern.

Secondary map-stability statistic:
- independently fit the selected-language emission map on two disjoint halves of the training folios;
- convert each surface unit to its maximum-posterior latent letter under `pi(letter)*E(letter,surface)`;
- compute frequency-weighted agreement across the two maps.

No latent plaintext sequence is emitted to the result files.

## Synthetic qualification families

Fresh pseudo-folios are generated from held-out language corpora and the source-grounded Amadi CYCLE homophone schedule.

Positive controls:
1. `BAV_GLOBAL`: Bavarian, one reusable key across all pseudo-folios.
2. `GER_GLOBAL`: German, one reusable key across all pseudo-folios.
3. `BAV_GLOBAL_SWAP`: Bavarian, one reusable key, with preregistered sparse local adjacent transpositions (5% of eligible adjacent same-type positions) after encryption.

Negative / adversarial controls:
4. `BAV_FRESH`: genuine Bavarian plaintext but an independently generated key per pseudo-folio. This is the critical control separating 'language ciphertext' from 'reusable manuscript-wide key'.
5. `GER_FRESH`: analogous German fresh-key control.
6. `STABLE_MARKOV`: a non-language stable typed first-order surface Markov process estimated from a separate same-key synthetic ciphertext; symbol identities are globally stable across pseudo-folios. This is the critical control against mistaking ordinary reusable surface grammar for a cipher key.

All controls use the same 144-symbol typed surface inventory and matched pseudo-folio/segment length geometry.

## Q0 calibration and untouched replication

Two deterministic namespaces are used: `CAL` and `VAL`, with disjoint plaintext spans, keys, pseudo-folios and random seeds.

Each namespace contains 3 independent replicates of each family. Each replicate uses four pseudo-folio folds; replicate statistics are the median fold ITE and median fold map stability.

Calibration succeeds only if:
- all 9 positive replicates have ITE > all 9 negative/adversarial replicates;
- all 9 positive replicates have stability > all 9 `FRESH` replicates;
- `STABLE_MARKOV` is below the ITE positive range.

If separable, freeze:
- `TAU_ITE = midpoint(min positive calibration ITE, max negative calibration ITE)`;
- `TAU_STAB = midpoint(min positive calibration stability, max fresh-key calibration stability)`.

Untouched VAL qualification then requires:
- >= 8/9 positive replicates pass both thresholds;
- 0/9 negative/adversarial replicates pass both thresholds;
- both `BAV_GLOBAL` and `GER_GLOBAL` each pass >=2/3 replicates;
- `BAV_GLOBAL_SWAP` passes >=2/3;
- no `STABLE_MARKOV` replicate passes both gates.

Any failure closes v6 before Voynich FIT scoring. Thresholds may not be relaxed after VAL.

## Q1 — Voynich FIT cross-folio test

Only after Q0 passes:
- partition the 181 FIT folios deterministically into 6 balanced hash folds;
- for each fold, fit on the other five folds and score the held-out fold;
- compute ITE against 24 topology-preserving per-folio permutations;
- compute map stability from two disjoint halves of the training folios.

FIT cipher-fingerprint gate:
- median fold `ITE >= TAU_ITE`;
- at least 5/6 folds `ITE > 0`;
- median fold `STAB >= TAU_STAB`;
- at least 5/6 folds have `STAB >= TAU_STAB`;
- the train-selected language must not alternate pathologically: one language must be selected in >=4/6 folds. This is diagnostic of a coherent latent model, not a Bavarian claim.

If FIT fails, C1 remains sealed.

## Q2 — C1 confirmation

Only if Q1 passes:
- fit once on all 181 FIT folios;
- select Bavarian/German on FIT train likelihood only;
- freeze language and emission map;
- score C1 once;
- compute C1 ITE against 64 topology-preserving per-folio permutations;
- compare FIT-half map stability as the pre-existing stability estimate; no C1 refit is permitted.

C1 confirms a reusable-key cipher fingerprint iff:
- `C1 ITE >= TAU_ITE`;
- observed C1 score exceeds every one of the 64 permutation scores;
- selected language is the same as the >=4/6 plurality language from Q1.

A confirmation is **cipher-interesting**, not a decipherment. A Bavarian-specific claim additionally requires Bavarian to be the frozen selected language and independent language evidence beyond the topology result; v6 alone cannot override the v5 language-vs-null failure.

## Stop rules

- No threshold changes after CAL.
- No target-driven increase in HMM starts/iterations.
- No per-folio key fitting on Voynich.
- No Currier/hand-specific keys in v6; those require a separately preregistered successor if v6 motivates them.
- No plaintext inspection.
- If a qualification gate becomes irrecoverably impossible, cancel remaining paid compute.
- Verify no Hugging Face job remains running at closeout.
