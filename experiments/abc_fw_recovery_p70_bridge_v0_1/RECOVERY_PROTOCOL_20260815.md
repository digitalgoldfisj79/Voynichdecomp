# ABC/FW recovery + P70 bridge v0.1

Date frozen: 2026-08-15, before any recovery/bridge statistic is computed on this branch.

This is a **recovery and mechanistic follow-up**, not a new independent discovery sample. The August 13 ABC/FW hypotheses and thresholds are not changed. The supplied August 13 JSON values remain the frozen historical outputs.

## Inputs

- `enriched_records.json`: ZLZI/P70 occurrence records, preserving folio, line and token order.
- Frozen FW carrier set from `FW_20260813.json`: `daiin, ol, chedy, aiin, cshedy, chol, or, ar, chey, dar, qokeey, qokeedy, cshey, qokain, qokedy, dy, qokaiin, al, dal, chor`.
- Primary tight null: independently permute the exact occurrence-record multiset within each line. No cross-line movement.
- 200 permutations, seed 20260813. This scale is chosen to recover the historical ABC output, whose reported null means have 0.005 granularity.

## R1 — ABC recovery

### A. Two legs
For each folio with at least 40 adjacent-pair opportunities, compute observed ED1 and exact lag-2 E2 counts and subtract their within-line permutation means. Spearman correlation of the two excesses is reported with p-value. Repeat after aggregating by P70 section. Preserve the original decision rule: one process only if rho >= +0.3 and p < .05; two-process evidence if |rho| < .15 or rho < 0.

### B. Crowding
For each adjacent ED1 event, classify by mean pair token length: short <=4, mid >4 and <=6, long >6. Compare each count to its same-null permutation distribution. Preserve the original crowding-falsification gate: long ratio >=1.15 and z>=2.

The historical preregistration named a matched `ReM` control but did not identify its bytes or construction in the supplied artefacts. It will not be silently replaced. If an exact ReM source is not recoverable, the result will be marked `REQUIRED CONTROL UNRECOVERED` rather than fabricated.

### C. Direction
For every permutation, recompute the **ratios** required by the original preregistration, rather than inferring them from marginal count SDs:
- accretion/reduction ratio;
- first-half/second-half substitution-site ratio.
Report observed ratio, permutation mean/SD, z and two-sided empirical p. The original directional gate is |z|>=2. Marginal category counts are retained for reconciliation.

## R2 — FW completion

The frozen carrier set is not reselected.

For every exact within-line `A B A` where A is a carrier, collect B. Report per carrier when n>=30 and pooled over all carrier closures regardless of individual power. Entropy fraction is H(B)/log2(n), with 0 for n<2, matching the stated maximum possible given B-slot count. Preserve the original gates: >=0.80 is nearly-free/function-word-compatible; <0.50 is repeated-phrase-like.

The August 13 positional-entropy and collocational-breadth values are treated as frozen historical results. Because the phrase `frequency-matched in bands` does not specify an exact matching algorithm, no reconstructed control selection may overwrite them. A nearest-frequency, without-replacement control set from ranks 40–200 may be reported only as a sensitivity analysis and will be labelled post hoc.

## R3 — prospective P70 mechanistic bridge

These tests are registered after seeing ABC/FW, so they are mechanistic follow-ups rather than independent confirmation.

### P70-ED1
For every adjacent ED1 event, compare the P70 parses of its endpoints and classify which slots differ: prefix, gallows, core, suffix, or multi-slot. Compare category counts to the same within-line permutations. Also classify endpoint core-state as both-empty, mixed, or both-nonempty.

Interpretation: concentration of ED1 excess in a specific slot-change class would localise the surviving local derivative process; a diffuse multi-slot excess would argue against a simple slot-edit mechanism.

### P70-E2
For each exact lag-2 closure, classify the repeated endpoint by P70 empty-core state and by membership in the frozen top-20 carrier set. Compare to the same tight null. Report positive-excess shares.

Interpretation: excess concentrated in empty-core carrier tokens would support the post-ABC hypothesis that the lag-2 phenomenon is recurrence of structural/template states rather than ordinary natural-language function words. Failure would reject that proposed bridge.

### Carrier P70 profile
Report empty-core fraction and dominant P70 parse for each frozen carrier and compare aggregate carrier empty-core frequency with the whole corpus and the post-hoc nearest-frequency sensitivity controls.

## Audit rules

1. Do not alter August 13 thresholds after seeing recovery results.
2. Report reproduction discrepancies against the supplied ABC JSON explicitly.
3. Preserve negative and unresolved outcomes.
4. Do not describe P70 bridge results as decipherment or proof of semantic content.
5. Do not substitute an unidentified ReM corpus.
