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

Reproduce the exact Q3 K=22 Arabic control under namespace `M19STAv17Q3`, including the same control span, synthetic value draw, opaque surface assignment, 45k/39k split, objective, legal K=22 constraint and frozen Arabic LM. Record the true and fitted maps and all mismatches. Reproduction must recover exact-map accuracy `0.7632051282051282` and independent-fit agreement `1.0` before diagnosis proceeds.

## D1 — optimizer check

Compare true-map and fitted-map objectives on the original 45k fit sample. Then run 24 independent restarts × 100,000 proposals, exact full scoring, followed by exhaustive legal pair-swap/single-move polish. If stronger search reaches the true map or exceeds the Q3 fitted objective by >=1e-5 nats/scored event, classify `OPTIMIZER MISS`; otherwise continue.

## D2 — state-signature geometry

For each of the 19 BnF numerical states, construct its observable Arabic-induced signature from outgoing transitions, incoming transitions, start and end probabilities. Standardize components, rank all 171 pair distances, and locate every mismatched true→fit pair. Enumerate exact single-transposition automorphisms at tolerance 1e-12.

## D3 — counterfactual likelihood

Evaluate the true, Q3-fitted and stronger-search maps by exact letter-HMM forward likelihood on the original 39k holdout and on fresh deterministic Arabic synthetic streams of 100k, 500k and 2M letters. Include 500 random legal K=22 maps as a null reference. Maps are fixed; no refitting occurs.

- If true increasingly dominates and beats fitted by >=1e-4 nats/letter at 2M: `FINITE-SAMPLE AMBIGUITY`.
- If fitted remains within 1e-4 nats/letter at 2M: `STRUCTURAL/NEAR EQUIVALENCE`.

## D4 — equivalence-aware metric development

Only after `STRUCTURAL/NEAR EQUIVALENCE`, define equivalence neighbourhoods from control-side state geometry and 2M counterfactual likelihood only. Report exact and equivalence-aware recovery. This cannot amend v1.7 automatically: any v1.9 must be prospectively frozen and rerun K=22/26/36 controls under a fresh namespace before any Voynich scoring.

## Verdict vocabulary

- `OPTIMIZER MISS`
- `FINITE-SAMPLE AMBIGUITY`
- `STRUCTURAL/NEAR EQUIVALENCE`
- `UNRESOLVED IDENTIFIABILITY FAILURE`

No Voynich score, decoded string, candidate language result, or RF/H17/C17 map may be generated or inspected in v1.8.
