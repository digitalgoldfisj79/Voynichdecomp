# Q1 prereg — blind recovery of SELECT-equivalence categories (2026-10-19)

Status: frozen before blind Q1 outputs. Synthetic ONLY. Parent `PROGRAMME_PREREG.md` and `RUNNING_RESULTS.md` govern interpretation.

## Specific new test
The Q0 experiment gave an optimistic **oracle** source-item NMI median of .626/.671 for weak language-like at 32/64 source types; .881/.931 for strongly sequential notation; table .535/.488; source-neutral zero. This does NOT identify source mechanisms in Voynich. Next measure the recoverability of the **eight SELECT categories**, not words, *blindly* through the existing K12/8-group/57-piece frozen FORM renderer.

## Inputs / forward model
- Frozen renderer SHA `16050050c11c41d222902668e84fa6e185a589c03be9a78b19529258f4b653de` (see Q0 script) with FORM piece legality, START and exact STOP hazards unchanged. Both ENTRY and weak ROUTE use the archived socket.
- Same 4 Q0 source families: LANG abstract weak sequence, NOTATION strongly constrained motif, TABLE source IID, NONE source-independent output. **Do not** interpret synthetic family names as historical Voynich classification. Family generation itself is unchanged.
- Eight controller states; each 32- or 64-member source vocabulary type deterministically maps to a controller, with multiple source types per controller. The blind instrument sees **neither** this map nor source transitions A, U/Ve/Vr, true family or source labels.
- 2,000 tokens per independent planted instance: 1,200 TRAIN / 400 VAL / 400 untouched TEST, each sampled as a fresh chain from the same generator with reset at boundary (prevents temporal fold leakage).
- Only synthetic FORM in Q1; CONNECT and LINE_ENTRY are not yet installed. No Voynich data are scored.

## Blind inverse and model comparison
- Fit an eight-state hidden Markov model whose emission family is **exact** frozen FORM socket: each class learns additive START bias over 12 legal classes and one token-constant destination ROUTE bias over the 12 legal destinations, shared across all 8 K8 source continuation groups. Exact-piece and probabilistic END laws are frozen identically under every candidate.
- Training uses forward-backward EM; weighted emission parameters optimised under exact normalized softmax with regularisation. Fit IID eight-state mixture baseline with identical emission capacity, different dynamics.
- Frozen candidate bank: `mode in {HMM, IID}`, regularisation `{0.3, 1.0}`, restart seeds `0,1,2`. Select using VAL average exact marginal log likelihood, with zero access to synthetic truth. If HMM wins, report predictive held-out gain relative to the best validation-selected IID model. Frozen 8 classes and stopping iteration cap; no per-instance refits triggered by true labels.
- Evaluation after selection: on untouched TEST only, posterior MAP NMI and ARI vs *true SELECT controller group* (label permutation invariant), matched truth-independent same-category posterior/pairwise AUC and true source-type equality AUC, exact marginal heldout HMM vs IID bits/token, and source-neutral negative false positives under frozen score thresholds.
- Since source type may collide in SELECT controller, raw source-token identity recovery is **not** the primary Q1 target. Independent section/physical folds and real Voynich come later.

## Primary gates
1. Instrument recovers planted controller categories in >=90% of 20 fresh seeds with TEST NMI >=.60, 10th-percentile NMI >=.60 and median NMI >=.70.
2. Wrong-family classification must not be inferred from an HMM-vs-IID win alone; only dynamic signal claims get confirmation if heldout gain >= .02 bits/token in >=90% relevant positives AND <=5% HMM false detections in source-neutral/table matched negatives. This .02 threshold is fixed upfront, NOT chosen from Q1 outcomes.
3. Heldout pairwise same-class discrimination AUC >=.80 on positives; source-type same-identity pair AUC separately exploratory.
4. If a gate fails, stop promotion and distinguish optimiser/power from actual source-family failure. Underpowered family is marked VOID, not falsified.

## Stages
- 2 pilot/discovery seeds per family at 32 types to verify plumbing; then new 20-seed confirmation panel (at 32 + 64 types only if pilot runs).
- No parameter truth, source family or real Voynich dictionary data in training/validation selection.
- Numerical exact likelihood correctness: test zero-bias source distributions against reference `token_loglik`; fail if tolerance >1e-8.
- Report total runtime and source audit.
