# SELECT -> state-separated FORM interface specification v1
Date: 2026-10-04
Status: PREREGISTERED DESIGN
Scope: define the input shape presented by SELECT to the frozen FORM renderer.
Firewall: NO P70. Synthetic calibration only until recovery passes.

## Motivation

The previous inverse-renderer calibration let one latent source item bias an arbitrary low-rank distribution over the full 57-piece transition surface of a token.

That factorisation is too permissive because earlier FORM work already established that most within-token structure belongs to FORM itself:

- stable ~32-piece repertoire;
- context-sensitive STOP/CONTINUE hazard;
- continuation compatibility compressible to ~8-10 source-response states;
- ~69 continuation compatibility edges in the strict K8 representation;
- short 1-/2-state cycles arise from local FORM dynamics;
- Mauro LOOP/slot systems are surface projections of this state process rather than independent token generators.

Therefore SELECT must not be allowed to reinvent token grammar.

## Frozen FORM socket

FORM owns:
1. the learned piece repertoire;
2. legal START pieces/classes;
3. current visible piece and its FORM class;
4. token depth;
5. incoming FORM class/state;
6. STOP/CONTINUE hazard;
7. legal continuation topology;
8. continuation response states (~8-10; K8 strict primary);
9. exact-piece realization conditional on the legal FORM route;
10. short cycles that arise from legal transitions.

LINE_ENTRY and CONNECT remain separate frozen modules.

## SELECT output

SELECT supplies a control signal to FORM. It does not output a token and does not carry its own transition grammar.

### Socket A — ENTRY (mandatory)
For token t, SELECT may bias which legal FORM entry is chosen.

Primary representation:
- bias over legal START K12 class / exact initial-piece alternatives;
- normalized only over FORM-legal START choices;
- no new start symbol or edge may be created.

This is the minimum interface and corresponds to the existing empirical result that token/class initiation remains the largest unresolved SELECT term.

### Socket B — ROUTE preference (optional, tested after A)
A token-constant control vector may remain active after entry and bias destination FORM classes among already-legal continuation edges.

Formal form:

P(dest | current_FORM_state, SELECT_t)
proportional to
P_FORM(dest | current_FORM_state) * exp(b_t[dest])

subject to:
- the frozen legal edge mask;
- same b_t held constant for the whole token;
- no current-state-specific transition matrix supplied by SELECT;
- no new edge;
- no modification of FORM state identity.

This is an attractor/preference signal, not a second grammar.

Primary destination coordinate is frozen K12 destination class, while row behavior is governed by the state-separated ~K8/K10 source-response equivalence structure.

### Socket C — exact-piece preference (not initially licensed)
SELECT may bias exact piece identity within a destination class only if A+B fail a preregistered residual test.

It cannot alter destination legality.

### Socket D — direct STOP intention (hostile/last resort only)
No direct SELECT->STOP parameter is allowed initially.

STOP already has strong heldout dependence on:
current piece + depth + incoming FORM class.

A direct SELECT stop/depth control can be introduced only if a prospective residual test shows unexplained source-conditioned termination after A+B+C.

## Forbidden SELECT degrees of freedom

SELECT may NOT:
- supply its own 57-piece transition matrix;
- alter the FORM legality graph;
- create new piece sequences outside frozen FORM;
- assign arbitrary per-edge weights conditional on hidden state;
- redefine STOP/END behavior in the primary model;
- encode LINE_ENTRY behavior;
- encode CONNECT boundary behavior.

## Synthetic recoverability programme

The synthetic generator must now use the frozen state-separated FORM socket itself.

Hidden source item X_t is planted upstream of SELECT.

Nested planted families:

F0 ENTRY only.
F1 ENTRY + token-constant ROUTE preference.
F2 ENTRY + ROUTE + within-class exact-piece preference, only after F1 calibration.
F3 direct STOP control only as a diagnostic extension.

For every family:
X -> SELECT control -> frozen FORM -> frozen CONNECT/LINE_ENTRY context as applicable -> surface.

Inverse search sees only the synthetic surface and the frozen FORM socket definition.

Final scientific scoring uses the exact same forward family as generation.

## Primary discrimination questions

1. Is ENTRY-only rich enough to carry a recoverable hidden source?
2. Does adding token-constant ROUTE preference materially increase recoverable information?
3. Can F1 reproduce the residual families that arbitrary piece-channel inversion was trying to absorb?
4. Does the correct factorisation improve blind identifiability relative to the old arbitrary 57-piece source channel?
5. Is any direct exact-piece or STOP control actually necessary?

## Promotion rules

Prefer the narrowest socket that:
- passes renderer-native planted-source recovery under the frozen recovery contract;
- achieves heldout NMI >= .70 median with required replicate stability;
- preserves exact FORM legality and termination rules;
- improves the relevant heldout residual family;
- does not require a richer hidden transition grammar than the evidence supports.

No real Voynich inversion until at least F0 and F1 synthetic calibration are complete.

## Interpretation

This specification does not assume what the upstream content is.

It defines only the shape of the interface:

SELECT chooses a legal entry condition and, if required, a low-dimensional token-level preference over legal FORM trajectories.

The FORM automaton remains the renderer.
