# BnF M19 STA Identifiability v1.8 — Control-Only Protocol

Date: 2026-08-09
Parent: v1.7 Q3 result commit `15e1cfa0e37119907d6a99ba6b2e2be1c4730fa6`
Scope: **control-only**. No Voynich RF/H17/C17 language score may be generated in this experiment.

## Question

Why did the binding v1.7 Q3 K=22 Arabic control identify Arabic correctly with 100% independent-fit agreement but recover only 76.3205% of the hidden numerical map?

Three possibilities are distinguished prospectively:

1. **optimizer miss** — the fitted map is measurably worse than the true map under the frozen Arabic objective and stronger search recovers the true map;
2. **finite-sample ambiguity** — alternative maps are close on the 45k fitting sample but separate under much larger fresh synthetic samples from the same frozen Arabic generator;
3. **structural/near equivalence** — distinct legal maps have effectively indistinguishable induced Arabic likelihood because some BnF numerical states are observationally exchangeable or nearly exchangeable under the frozen Arabic LM/channel.

## Frozen reproduction

Reproduce the exact Q3 K=22 Arabic control under namespace `M19STAv17Q3`, including:

- same UD Arabic dev+test control pool;
- same first fitting-half-support-complete span selector;
- same synthetic numerical-value draw;
- same opaque-surface assignment;
- same 45,000 fitting / 39,000 held-out split;
- same generalized M19 objective and legal K=22 map constraint;
- same frozen Arabic LM and induced numerical pair model.

Record the true 22->19 map, the two Q3 fitted maps, surface frequencies, objective scores, and all mismatched surface states. Reproduction must recover the v1.7 Q3 mapping accuracy `0.7632051282051282` and independent-fit agreement `1.0` to numerical tolerance before diagnosis proceeds.

## D1 — optimizer check

Compare the true-map objective and fitted-map objective on the original 45k fitting sample. Then run a stronger control-only search:

- 24 independent annealing restarts;
- 100,000 proposals/restart;
- exact full-score evaluation;
- deterministic exhaustive 1-move and pair-swap polish to local optimum.

Interpretation:

- if a stronger search reaches the true map or a map with materially higher objective than the Q3 fitted map, classify `OPTIMIZER MISS`;
- otherwise continue.

A material objective difference is >=1e-5 nats per scored event.

## D2 — exact state-signature geometry

For each of the 19 BnF numerical states under the frozen Arabic induced model, construct its observable signature from:

- outgoing log-transition row;
- incoming log-transition column;
- start log-probability;
- end log-probability.

Standardize components by their empirical scale and calculate all pairwise distances/cosines.

For each mismatched true->fit state pair, report its rank among all 171 state pairs and whether it lies in the closest 1%, 5%, 10%, or outside.

Also enumerate exact automorphisms of the 19-state induced pair model under single transpositions: a transposition is exact only if swapping the two numerical labels leaves transition/start/end arrays equal within `1e-12`.

## D3 — counterfactual map likelihood

On the original 39k held-out control, compute exact letter-HMM forward likelihood under:

- the true map;
- the Q3 fitted map;
- the stronger-search map;
- 500 legal random K=22 maps.

Repeat on fresh independently generated Arabic synthetic streams of 100k, 500k and 2,000,000 letters, using the same frozen Arabic plaintext model but fresh deterministic seeds. The true map and candidate alternative maps are held fixed; no map is refitted on these streams.

Report per-letter likelihood gaps and the percentile of each map among random legal maps.

Classification:

- if the true map increasingly dominates with sample size and exceeds the fitted map by >=1e-4 nats/letter at 2M, classify `FINITE-SAMPLE AMBIGUITY`;
- if the fitted/alternative map remains within 1e-4 nats/letter of the true map at 2M, classify `STRUCTURAL/NEAR EQUIVALENCE`.

## D4 — equivalence-aware recovery metric development

Only if D3 returns `STRUCTURAL/NEAR EQUIVALENCE`, define a control-side equivalence-aware map score without using Voynich:

1. Build equivalence neighbourhoods from the frozen Arabic state-signature geometry only.
2. A predicted state counts as equivalent to the true state only when exchanging them changes the 2M exact forward log likelihood by <1e-4 nats/letter under the fixed synthetic control.
3. Report both exact recovery and equivalence-aware recovery.

No v1.8 result automatically amends v1.7. Any future v1.9 protocol that adopts an equivalence-aware gate must be frozen prospectively and rerun all K=22/K=26/K=36 controls from scratch under a fresh namespace before any Voynich score is unlocked.

## Verdict vocabulary

- `OPTIMIZER MISS`
- `FINITE-SAMPLE AMBIGUITY`
- `STRUCTURAL/NEAR EQUIVALENCE`
- `UNRESOLVED IDENTIFIABILITY FAILURE`

No Voynich score, decoded string, candidate language result, or RF/H17/C17 map may be generated or inspected in v1.8.
