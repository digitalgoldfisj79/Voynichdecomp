# Voynich context-first CF3 real lexical-equivalence graph — 2026-10-05

- key: `voynich_context_first_CF3_20261005`
- status: completed
- created: 2026-10-05T20:30:18.786756+00:00
- updated: 2026-10-10T16:27:25.988447+00:00

---

# CONTEXT-FIRST CF1/CF2/CF3 — REAL VOYNICH CONTEXT GRAPH
Date: 2026-10-05
Status: COMPLETE / PRIMARY GATE PASS
HF: 6ac40854404719ba37658ad5
GitHub: 23e2892e7123878fed7703c9d5f59d69562fd502
Script: research/context_first_CF1_CF3_graph_20261005.py

## Firewall
Strict +P0 ordinary running text; current physical line starts excluded.
Discovery folds2/3; validation4; final untouched0/1.
Current token spelling, FORM, ED, DINO, length, gallows, Stolfi/Mauro are invisible in graph discovery.
Only exact CURRENT token ID is a row key; graph is learned from EXTERNAL neighbour token identities at lags -2,-1,+1,+2, nuisance-residualized by current section and line-position class.
Eligible >=20 discovery occurrences, >=3 validation, >=3 final.
Context vocabulary top256 discovery only.

Discovery graph:
all token-pair edges above q99.5 of frequency+dominant-section matched random-pair independent-fold exchangeability statistic.
Final null preserves discovered graph topology/degree via matched node relabelling.

## ZLZI
population 27,120; eligible types103; q99.5 .146411; 13 edges.
Components:
1 {aiiin,aiin,ain,air,ar,or,r,s} size8
2 {chody,qotal}
3 {okal,otar}
4 {okar,otam}
5 {qokeedy,qokeey}
Validation similarity .189887 vs null .032053 SD .024778 => 6.370 SD.
Final reciprocal similarity .130156 vs null .007527 SD .016706 => 7.340 SD.
Fold0 within .147685; fold1 .120580, both positive.
Exact-token self positive control 15.931 SD.
PASS.

## ZLZB
Same 13 edges / same 5 components as ZLZI.
Validation 6.257 SD.
Final 7.204 SD.
Fold0 .147659; fold1 .120793.
Exact-token self 15.701 SD.
PASS.

## TTLI
13 edges; components:
{aiiin,aiin,air,al,ar}
{chcthy,otam,qokain}
{or,r,s}
{qokedy,qokeedy,qokeey}
{okal,otar}
Validation 4.358 SD.
Final 5.722 SD.
Fold0 .092304; fold1 .072028.
Exact-token self 10.521 SD.
PASS.

## Adjudication
A context-first lexical-equivalence graph exists prospectively and replicates across all three transcriptions. This directly overturns the overly broad inference from NK3 that no latent equivalence structure is detectable: NK3 rejected SURFACE-FIRST families, not context-first lexical equivalence.

This is NOT yet proof of homophony. The graph may capture lexical/functional paradigms, grammatical substitution classes, or other contextual equivalence. CF4 is required: infer class from external context with current token masked and ask whether class predicts the revealed surface structure of genuinely novel exact token types.

Important observation after freeze only: several ZL components are visibly related (aiin/ain/ar/or etc.; qokeedy/qokeey), while some pairs are not obviously close (chody/qotal, okal/otar, okar/otam). Surface inspection must be quantified post-freeze; it was not used to discover them.

