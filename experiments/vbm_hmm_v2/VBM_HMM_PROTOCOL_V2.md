# VBM v2 — Typed Trigram-HMM Inference Protocol

Date: 2026-08-11
Namespace: `VBMHMMV2`
Parent closeout: `87b54965a59b0bc1c12d93c8e7311f30cde6f399`

## Motivation

VBM v1 produced a prospectively held-out C/V topology result that selected Bavarian over German and Italian, but its direct map-space annealer failed qualification because the highly homophonic Bavarian controls did not converge to a unique deterministic map. No typed Voynich H1 substitution fit was ever run.

v2 changes the inference family, not the representation: latent plaintext letters form an order-2 Markov chain under the fixed language trigram model and observed VBM units are emitted by typed homophonic emission distributions. Parameters are learned by Baum–Welch/EM. No annealing or direct symbol-map search is used.

## Frozen representation and data

Unchanged from VBM v1:
- 21 core surface units; core units may emit consonants only.
- 123 bridge surface units selected on FIT-A at >=0.995 occurrence coverage; bridge units may emit vowels only.
- `VR=qo` when the retained token begins `qo`; otherwise VR is the first retained character.
- VL is the final retained character.
- right-edge `eed` then `ed` are C2 composite core units; other maximal `e+` runs are composite core units.
- FIT-A = original 181 folios.
- VBM_H1 = `f28v f31v f88r f5r f34r f81v`.
- VBM_C1 = `f85r1 f53v f33r f10r f23r f111r` and remains sealed unless H1 passes all v2 candidate gates.

Language panel is fixed:
1. Bavarian (`bavarian-nlp/barwiki-20250720`) — primary hypothesis.
2. Standard German (UD German GSD) — nearest rival.
3. Italian (UD Italian ISDT) — source-native comparator.

Normalisation is unchanged: 19-letter `abcdefghilmnopqrstu`; j->i, v/w->u, y->i, x/z->s.

## Model

For each language L:
- hidden plaintext letters x_t follow the fixed trigram language model P_L(x_t | x_{t-2},x_{t-1});
- each observed surface unit y_t is emitted from the current hidden letter with typed emission probability E_L(y_t | x_t);
- core observations have zero emission probability from vowels;
- bridge observations have zero emission probability from consonants;
- each typed plaintext letter must retain non-zero mass over at least one surface unit.

EM estimates only E; language transitions remain fixed.

### Initialization and ensembles

Fresh deterministic seeds under `VBMHMMV2`. For each candidate language and control:
- 8 random Dirichlet initializations, split into ensembles A/B (4 each);
- Dirichlet concentration 0.35 over allowed surface units;
- additive emission pseudo-count 0.05 per allowed surface/plaintext cell;
- maximum 60 EM iterations;
- stop when relative log-likelihood improvement < 1e-7 for 3 consecutive iterations.

The best training-likelihood run per ensemble is retained.

### Hardened map

For each surface unit y, hard value = argmax_x posterior expected count N(y,x) from the final E-step. Surjectivity is repaired only if an allowed plaintext letter has no assigned surface: choose the reassignment with minimum posterior-count loss. This rule is deterministic.

A/B map agreement is occurrence-weighted over the fit ciphertext.

## Q1 — fresh positive controls

Fresh hidden maps and fresh spans are generated under the new namespace; no v1 hidden keys or spans are reused.

For each language, 4 replicates:
- fit plaintext >= 40,000 normalized characters;
- holdout plaintext >= 15,000 normalized characters;
- encrypt with the frozen 21+123 typed surface inventory and a fresh deterministic surjective homophonic map; surface choice within each plaintext letter is uniform across its assigned homophones.

For each replicate fit all 3 candidate language HMMs.

A replicate qualifies only if:
- correct language ranks first on fixed-emission holdout log-likelihood;
- margin over runner-up >= 0.02 nats/event;
- correct-language hidden-map recovery on holdout >= 0.85;
- A/B occurrence-weighted hardened-map agreement >= 0.90;
- A/B holdout score gap <= 0.01 nats/event;
- both retained A/B EM runs satisfy the convergence stop.

Q1 family gate:
- 12/12 correct language rank1;
- >=11/12 replicates individually qualify;
- each language >=3/4 individually qualify;
- median recovery >=0.95;
- minimum recovery >=0.85.

Language-specific absolute H1 floor = 5th percentile of the four correct-language holdout scores. With four replicates, this is intentionally conservative and interpolation-based.

## Q2 — structured negatives

Only if Q1 passes.

50 negatives: 10 each IID, order-2 permuted Markov, motif, copy/mutate, slot grammar. Each preserves typed core/bridge status and approximately the FIT-A type-frequency profile but has no natural plaintext.

Each negative is fit under all three language HMMs. It is a false positive only if:
- selected language score >= that language's Q1 floor;
- selected-over-runner-up margin >=0.02;
- A/B agreement >=0.90;
- A/B score gap <=0.01;
- both ensembles converged.

Q2 gate: <=1/50 total false positives and <=1 for any generator class.

## H1 target

Only if Q1 and Q2 pass.

For each language:
- fit E on FIT-A only;
- score VBM_H1 without refitting;
- retain A/B independent estimates.

A language is an H1 candidate only if:
- H1 score >= its frozen Q1 absolute floor;
- A/B H1 score gap <=0.01;
- A/B occurrence-weighted hardened-map agreement >=0.90;
- both ensembles converged;
- it ranks first on H1 with margin >=0.02 nats/event.

Primary Bavarian candidate additionally requires Bavarian rank1.

If no language passes: no C1.
If a candidate passes: open VBM_C1 once and apply the already fitted A/B emissions without refitting. Confirmation requires the same absolute floor, A/B, and rank-margin gates on C1.

## Stop rules

- No changing VBM unitisation, bridge vocabulary, language panel, LM order, smoothing, number of starts, EM threshold, or candidate gates after any H1 HMM score is produced.
- If Q1 fails, H1 is not scored.
- If Q2 fails, H1 is not scored.
- If H1 fails, VBM_C1 remains sealed.
- A further v3 would require another materially new inference family, not additional EM starts or iterations.
