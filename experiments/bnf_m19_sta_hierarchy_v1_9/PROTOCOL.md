# BnF M19 STA/aaa Hierarchy v1.9 — Strong-Optimizer Confirmation

Date: 2026-08-09
Parent diagnosis: v1.8 `OPTIMIZER MISS`, commit `ed426f774200d756bd09f979d75a9e14435df368`.

## Purpose

Rerun the STA/aaa hierarchy from scratch under a fresh namespace with a convergence-controlled optimizer strong enough to avoid the v1.7 K=22 Arabic local optimum. No threshold is weakened.

## Frozen representations and hierarchy

The v1.7 source files, hashes, parser, official Zandbergen bitrans conversion, RF 60/20/20 folio split, and three representations are unchanged:

- STA family vocabulary: K=22;
- connected `aaa` vocabulary: K=26;
- full STA vocabulary: K=36.

The candidate plaintext panel remains Latin, Italian, German, French, Greek, Hebrew, Arabic and Spanish. Positive-control languages remain Latin, Italian, German, French, Arabic and Spanish.

## Fresh qualification namespace

All binding synthetic qualification data are regenerated under `M19STAv19Q1`. Candidate control spans come only from the same untouched UD dev+test pools used in v1.7 and must have fitting-half support for all 19 BnF values.

No v1.7/v1.8 fitted map is reused.

## Adaptive convergence optimizer

Every fit consists of two independent ensembles A and B. Each ensemble runs batches of six annealing restarts. Each restart uses:

- 100,000 legal proposals;
- exact full-score evaluation;
- the v1.8 temperature schedule;
- exhaustive legal pair-swap and single-move polish.

After each paired batch (6, 12, 18, 24 restarts per ensemble), stop only when:

1. best objective scores A/B differ by <= `1e-7` nats/event; and
2. occurrence-weighted map agreement A/B is >=0.95.

Maximum budget: 24 restarts per ensemble (48 total). If convergence is not reached, that fit fails instrument qualification.

For synthetic controls only, the known hidden map provides an additional oracle diagnostic: each ensemble best score must be no worse than the true-map score by more than `1e-6` nats/event. This oracle is never available or used on Voynich.

## Qualification gate

At each K=22/26/36, all six controls must satisfy:

- correct language rank 1;
- language margin >=0.05 nats/letter;
- median exact map recovery >=0.95;
- minimum exact map recovery >=0.85;
- independent ensemble map agreement >=0.90;
- optimizer convergence reached;
- both ensemble objective gaps to the true-map objective >= -1e-6.

All three K gates must pass before any RF H19 language score is generated.

## H19 / C19 hierarchy

If qualification passes, fit all eight candidate languages separately on RF T19 for each representation using the same two-ensemble adaptive optimizer, without any oracle information.

H19 gate at each representation:

- coverage >=0.97;
- top language rank 1 with margin >=0.05;
- top-language A/B map agreement >=0.90;
- optimizer convergence reached.

The hierarchy passes only if family, aaa and full-STA all pass and choose the same top language.

Only then unlock RF C19. C19 confirmation requires, for every representation:

- same frozen candidate rank 1;
- margin >=0.05;
- coverage >=0.97;
- same candidate top in all four deterministic C19 folio buckets with positive margin.

Only after RF C19 confirmation may the frozen maps be transferred without refitting to independent IT, ZL and GC STA streams, using the v1.7 coverage/short-space rules.

## Verdict vocabulary

- `STA/AAA STRONG INSTRUMENT NOT QUALIFIED`
- `NO STA/AAA M19 SIGNAL`
- `REPRESENTATION-SENSITIVE / NO HIERARCHY SIGNAL`
- `H19 STA/AAA CANDIDATE / C19 FAILED`
- `CONFIRMED STA/AAA M19 SIGNAL <language>`
- `CONFIRMED RF / TRANSCRIPTION REPLICATION FAILED`
- `CONFIRMED RF + INDEPENDENT TRANSCRIPTION REPLICATION`

A statistical signal is not a plaintext or language-identification claim without later historical-language and semantic validation.
