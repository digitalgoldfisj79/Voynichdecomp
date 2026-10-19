# CORRECTION 2026-10-19: No agreed plant name is required

**Supersession:** This document withdraws as an impossible/invalid promotion gate the requirement in `IDENT_GATE_RESULTS_20261019.md`, `IDENT_GATE_PREREG_20261019.md` and `anchor_candidate_gate.py` that a Voynich illustration first receive a verified botanical *species name* or an agreed VMS text-to-plant label. No Voynich botanical taxon is to be treated as agreed merely because an archival database or model assigns it. The existing f2r/centaury assay is retained as **post-selected surface-form discovery**, not a verified plant-name bridge; the independent medieval centaury *witness* establishes that such historical labels existed, not a VMS reading. The manuscript's semantics cannot be used as a prerequisite for decoding itself.

## Identifiability target

The investigable question is **does the source SELECT carry stable, repeatable information *beyond* FORM/CONNECT/LINE_ENTRY and defensible production-memory explanations?** This differs from the unanswerable question "what botanical species does a Voynich page depict?"

### Registered competing explanatory mechanisms

M0 **conditional generator:** SELECT transitions are generated from observed section, scribe, line entry, position, and prior visible FORM context, with no durable latent referents.

M1 **production/repertoire memory:** same as M0 plus latent session/packet/bifolium-level state and copying-induced reuse of entire forms or 1-edit neighbourhoods. This is the HARD NEGATIVE, not an IID shuffle.

M2 **content-conditioned source:** same emission/rendering capacity as M1, plus persistent item/relation information that must transfer across genuinely separated production units in a section. "Content" can be recipes, arbitrary nonlinguistic code, or language; no linguistic/semantic label is attached.

M3 **lexical source:** only after M2 beats both controls, assess structured natural-language alternatives with period-appropriate plaintext positive controls and explicit model recoverability. English/Italian/Latin dictionary matches across many languages are not a lexical proof.

Predictions must be formulated *before* inspecting held-out outcomes: for example, M2 gains in previously untouched physical bifolia, even when session-local burstiness and one-edit copying are represented at equal model capacity; M1 models session-local adjacency but makes no improvement on independent source-item transfer. This may still not separate all conceivable copying schemes; if a production model is observationally equivalent to M2, record that limit.

### Required study design (no named botanical anchors)

1. **Keep ecologies separate.** Herbal-A and Herbal-B *running text*, Q13 Balneological *running text*, Recipes/Stars *running text* have independent models/pools. Pharmaceutical is label text and never mixed; circular/spatial are distinct.
2. **Strict physical folds.** All sides of each bifolium remain in the same fold. Control section, scribal hand/Currier, quire, text lengths, page/line location and corpus/transcriber; don't trust arbitrary page-level discovery/validation flags until checked. Do not assume bifolio sequence is reading order.
3. **Frozen FORM/CONNECT/LINE_ENTRY.** Reuse established empirical rules and calibrated uncertainties. If a full generator isn't implemented, treat the experiment as a *component-level synthetic qualification only* and don't call it a full Voynich test.
4. **Positive calibration:** authentically sourced historical Latin/Italian/German prose and structured recipe/notation controls encoded through proposed mechanisms at comparable inventory, line lengths, morphology and source entropy; score blind inverse or prediction after locking hyperparameters. Source mechanism must be recoverable >=90% and hard-negative false-positive <=5%; 20+ independent instances minimum. Drawn from periods/genres possible in 15th century, record witnesses/citations.
5. **Hard negatives:** matched session memory, word-family one-edit reuse, position-conditioned graphotactics, section-conditioned vocabularies, and high-order observable context. Compare exact held-out cross-entropy and recurrence precision at a prespecified false-positive rate; AUC alone insufficient.
6. **No semantics implied.** A successful M2 identifies *unexplained transfer/persistent source information* not "plants have names" or "Voynich is a language". M3 requires separate multilingual/period-text recoverability and document anchors.
7. **F2R exploratory example:** ZLZI f2r line1 `kydainy` vs line8 `kydain`, edit distance 1; 0/28 matched other Currier-A Herbal line1/line8 control folios as close. Already post-selected by earlier researchers; no p-value or plant reading. Not eligible as confirmation.
8. **Review output:** for each model give identifiable observables, equivalent alternatives, validation/holdout sizes, context-preserved negative, positive recovery power/FPR, and predicted difference under an observable intervention. Explicit FAIL/VOID states, no invented meanings.

## Implementation correction

The previous `anchor_candidate_gate.py` is a *historical named-anchor intake validator* only, and **must not be used to block label-free mechanism tests**. No task depends on someone agreeing a Voynich botanical identification. Do not resurrect it as a global gate.

This correction is deliberately scope-limited: it does not claim that M2 is distinguishable from all possible production mechanisms, or that existing phase Q0–Q1B gates have passed. First confirm genuine discriminating power and then apply to real text.
