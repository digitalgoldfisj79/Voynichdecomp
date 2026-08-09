# Tranchedino × STA v2.3 — B1 implementation freeze

Date: 2026-08-09
Status: **FROZEN BEFORE Q1**

Binding solver file:
`experiments/tranchedino_sta_v2_3_stageb/b1_solver.py`

GitHub implementation commit:
`fd225442d3e84950bc997f405b9d6552e4965873`

Local execution copy SHA-256:
`2308e6ce5519829e0f59b5590dc1971a2ce6776d0596588b18065ee99034083a`

Binding source SHA-256:
`c5eba63cbe8055d3506d099043f5df23fd427df709546df6de70e084fedd3cf6`

## Frozen search architecture

1. Build the frozen Paduan line-reset alpha=.5 character quadgram/unigram model from the pre-page-183 source.
2. For each of four nuisance null-rate proposals (.01,.03,.06,.10), construct a frequency-based K92 initialisation using source-only proposal frequencies.
3. The inferred surface partition is reduced only to **alphabetic vs residual**. The 36 inferred alphabet signs stay in that partition during polishing. The remaining 56 signs may interchange geminate, null and nomenclator semantics.
4. Search objective: total frozen Paduan character log-likelihood per observed cipher event, subject to the preregistered 0.90–1.12 decoded-character/event constraint.
5. Coordinate polish allows pair exchanges within the two partitions and replacement of assigned nomenclator words by unused members of the frozen top-96 pool.
6. At most eight polish cycles per start.
7. Ensemble A uses the deterministic frequency start for each null-rate proposal.
8. Ensemble B uses one independently seeded bounded perturbation of each proposal start before the same polish.
9. Each ensemble returns the best of its four proposal starts by the frozen event-normalised search objective.
10. Recognition/generalisation scores are reported separately as nats per decoded character; the frequency proposal score is never used as manuscript evidence.

No D1 data may be used to modify this implementation after this freeze. Q1 decoded strings remain sealed; only numerical truth metrics are emitted.

## Early-stop rule

The Q1 protocol contains binding **minimum** recovery gates. Therefore, if any completed Q1 control has plaintext recovery <0.75, occurrence-weighted semantic recovery <0.70, nomenclator occurrence-word recovery <0.60, geminate occurrence recovery <0.70, nonzero-null F1 <0.75, held-out surface coverage <0.95, or fails the mandatory A/B convergence condition, the final qualification verdict is already negative. Remaining Q1 controls cannot repair that minimum and must not be run merely to complete a table.
