# Q13-informed SELECT source model specification v1
Date: 2026-10-04
Status: ACTIVE DESIGN
Scope: replace small latent-state calibration with section-conditioned structured source sequences.
NO P70. No real semantic decoding claim.

## Why this supersedes the small-state abstraction

The K8/K16 hidden-state calibration was only a recoverability scaffold. It is not a plausible model of the source.

Q13 provides the empirical source-shape clue:

1. Non-y morphology recovers true physical Q13 mates above chance.
2. The strongest mate affinities are packet-specific (especially f77/f82 and f79/f80), not a universal page-pair effect.
3. Contiguous vocabulary/morphology state predicts held-out Currier-89/dy abundance even when y/89 is withheld from predictors.
4. The broad morphology-state phenomenon is not unique to Q13, but Q13 has a strong local/section-conditioned realization profile (notably edy-rich within matched Currier B).
5. After SG47/SG64 family conditioning, the local y<->dy effect does not require a new SELECT operator: much of the visible alternation is downstream FORM-family realization.
6. External line-by-line source fits to Laufenberg / De Balneis fail. Q13 therefore supports structured source-side variation without licensing a particular plaintext.
7. Physical-mate affinity survives in non-y morphology, while traversal direction does not. This is compatible with packet-scale source/repertoire continuity passed through FORM, but does not distinguish source packet from production-session context by itself.

The correct source abstraction is therefore a structured section-conditioned sequence, not an 8-state controller.

## Architecture

source item x_t
+ section/register s_t
+ sequential context h_t
        |
        v
section-conditioned SELECT encoder
        |
        +--> strong legal FORM ENTRY preference
        +--> weak token-constant legal ROUTE preference
        |
        v
frozen FORM
        |
        v
surface Voynich token

FORM continues to own:
- piece repertoire;
- STOP/CONTINUE;
- legal continuation graph;
- continuation response states;
- exact-piece realization;
- short FORM cycles.

SELECT does NOT acquire its own grammar.

## Source families to calibrate

### L — language-like
- large reusable source vocabulary;
- Zipf-like type frequencies;
- section-specific lexical distributions;
- local sequential dependence (at least bigram/trigram or low-rank contextual model);
- repeated source items may recur in local packets;
- source identity is richer than the SELECT control vector.

### N — notation-like
- smaller event/symbol vocabulary;
- stronger transition constraints;
- repeated motifs / phrases;
- section-specific symbol/event distributions;
- potentially lower lexical entropy but stronger syntax/order.

### T — table/codebook-like control
- section-specific repertoire;
- weaker sequential syntax;
- item selection dominated by local table/repertoire distribution;
- included as a hostile structured non-language control.

These are source-shape controls, not historical claims.

## Q13-derived target phenomena

A viable source family should be able to produce, through the same frozen SELECT->FORM socket:

A. section-conditioned surface distributions;
B. local contiguous morphology/vocabulary states;
C. packet-scale / bifolium-scale affinity in source-conditioned morphology without requiring exact token identity;
D. stable global FORM legality across sections;
E. visible realization changes that can collapse after FORM-family conditioning;
F. no requirement for a universal small hidden controller;
G. recurrence/context information useful for inversion beyond the instantaneous FORM control vector.

## Recovery target

Do NOT demand recovery of arbitrary latent state labels.

Primary targets:
1. source-token equivalence / source-type identity where identifiable;
2. source sequence likelihood and heldout next-item prediction;
3. recovery of repeated source-item links;
4. section-conditioned source lexicon/repertoire;
5. pairwise source-identity posterior (same underlying item vs different);
6. source transition/context structure up to permutation/equivalence.

If two source items induce indistinguishable SELECT controls and indistinguishable sequence context, they are observationally equivalent and should not count as a decoding failure.

## First calibration ladder

Q0 plumbing:
Known structured source -> known SELECT encoder -> frozen FORM -> surface.

Q1 language-like, one section:
Test whether sequential context resolves source items that collide at the instantaneous SELECT socket.

Q2 language-like, multiple sections:
Shared FORM, section-conditioned source lexicon and source transitions.

Q3 notation-like, multiple sections:
Same renderer and same SELECT socket.

Q4 table/codebook-like hostile control:
Determine whether weak-syntax repertoire selection can reproduce the same recoverability / packet effects.

## Critical comparison

The key discriminator is not which source produces the most Voynich-looking tokens; FORM already guarantees that.

Compare source families on:
- recoverability of planted source relations;
- amount of sequential information surviving FORM;
- section conditioning;
- local state persistence;
- packet-scale affinity;
- recurrence structure;
- heldout source prediction.

## Real-manuscript bridge

Before any semantic inversion, derive the observable SELECT-socket trace from the real manuscript:
- ENTRY class;
- legal ROUTE trajectory counts / low-dimensional route summary;
- section;
- line/paragraph/page/bifolium packet coordinates.

Then measure which source-family calibration regime reproduces the real trace statistics, especially the Q13 packet/state phenomena.

Only a source family whose synthetic traces overlap the real manuscript on these preregistered diagnostics is licensed for later inversion.

## Adjudication

The K3c 20-seed small-state panel is retained as evidence that the ENTRY+weak-ROUTE socket can transmit recoverable information, but small hidden-state recovery is retired as a model of Voynich source content.

The research object is now:
structured source sequence -> section-conditioned SELECT -> frozen FORM.
