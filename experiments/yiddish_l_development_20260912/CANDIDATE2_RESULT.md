# Yiddish–German evaluator Candidate 2 — controlling development result

Date: 2026-09-12

**Status: `DEVELOPMENT_EVALUATOR_NOT_RESOLVED__STOP_AFTER_TWO_CANDIDATES`**

This is development-only. It is not an L certificate. The Voynich target was never loaded.

## Frozen-input provenance

Candidate-2 rules were frozen in `evaluator_candidate2_manifest.json` before outcomes. The pre-scoring corpus gate and extraction-code hash were committed separately in `candidate2_pre_scoring_gate.json` before language scores were computed.

German representation: ReF CorA-XML `tok_dipl/@utf` source/diplomatic layer; diplomatic fragments concatenated per virtual token; literal `a-z` filtering only after extraction; no `tok_anno`, lemma/POS, Unicode transliteration, or modernised reading text.

All pre-scoring gates passed. German BUILD comprised F014 and F015; DEVELOPMENT comprised F016, F018, F034, F037 and F148. BUILD-to-DEVELOPMENT overlap was zero exact 8-word types and zero 5-word types. Every German DEVELOPMENT family supplied >=512 words and BUILD supplied >10,000 words.

## Results

| arm | accuracy | effect over chance | exact balanced-label null SD | effect/null SD | exact one-sided p |
|---|---:|---:|---:|---:|---:|
| primary trigram | 1.000 | +0.500 | 0.166667 | 3.000 | 0.003968 |
| unigram nuisance | 1.000 | +0.500 | 0.166667 | 3.000 | 0.003968 |
| length-only nuisance | 0.900 | +0.400 | 0.163299 | 2.449 | 0.023810 |
| within-word-shuffled trigram nuisance | 1.000 | +0.500 | 0.166667 | 3.000 | 0.003968 |

Primary per-family calls were 10/10 correct. However, the frozen rule required the primary to outperform **each** shallow nuisance arm by at least one development-family error. It failed that requirement against both unigram and within-word-shuffle.

## Frozen decision criteria

- primary accuracy >=0.8: PASS
- primary effect >=2 null SD: PASS
- primary outperforms unigram by >=1 work-family error: **FAIL**
- primary outperforms length-only by >=1 work-family error: PASS
- primary outperforms within-word-shuffle by >=1 work-family error: **FAIL**

## Interpretation

The evaluator distinguishes the two corpus representations, but the result is not attributable to sequence-level language structure: equivalent separation persists under character unigram scoring and after deterministic within-word letter shuffling. The registered evaluator family is therefore unresolved after its two-candidate development limit and is closed rather than retuned.

This does not show that historical Yiddish and German are computationally indistinguishable, and it does not bear on Voynich directly. It shows that **this registered L evaluator cannot support a Yiddish-specific inference** under the current representations.

R remains the earlier finite-panel short-transfer recovery result. L is unqualified. T is not run. Voynich remains sealed.
