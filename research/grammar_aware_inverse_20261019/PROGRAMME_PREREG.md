# Grammar-aware inverse programme (GAI), stage 0 — preregistered 2026-10-19

**Status:** prospective instrument qualification only. Historical/plaintext conclusions forbidden until the exact gate passes. This programme extends, rather than replaces, `research/SELECT_FORM_INTERFACE_SPEC_20261004.md`, `research/INVERSE_RECOVERY_CONTRACT_20261004.md`, `research/Q13_INFORMED_SELECT_SOURCE_SPEC_20261004.md`, `research/VMS_RESIDUAL_RECOVERY_LADDER_20261008.md` and the previous [f75r exhaustive word-first rung](https://github.com/digitalgoldfisj79/Voynichdecomp/blob/experiment/f75r-progressive-cipher-ladder-20261018/research/f75r_cipher_ladder_20261018/RUNNING_RESULTS.md).

## Exactly what brute force established — and didn't
- Exhausted 26P8 = 62,990,928,000 injections of eight selected EVA glyph units into 26 plaintext letters; two equally exhaustive but weak positional-structure-breaking scrambled controls; fixed-key 8/25 novel f76r types accidentally matched at least one of 20 language dictionaries.
- Demonstrated *search throughput and apparent word matches*, **not** a source language or decipherment. K9–K20 was beam-pruned and the early glyph rung inventories differed.
- Real-over-scramble dictionary difference is confounded by positional FORM and CONNECT graphotactics. Never use global dictionary-hit count as the source evidence.
- Raw EVA glyphs are not identical to FORM pieces: the frozen grammar uses multigraphs and context-conditioned stops/transitions.
- Funding: prior 58 T4 jobs, est. $5.14 from elapsed times at $0.40/hr, NOT verified invoice/remaining credit.

## Factorization and scope
`source identity X_t (may be absent) -> SELECT (legal ENTRY, optionally weak token-constant ROUTE bias) -> frozen FORM (piece grammar, exact piece choice, STOP hazard) -> CONNECT / LINE_ENTRY (frozen context treatment) -> visible token`.

FORM owns legality, exact piece, depth-dependent END and short cycles. SELECT may not learn an independent arbitrary token grammar. Context modules may not depend on hypothetical plaintext. Section, scribe, physical bifolium and paragraph registers are explicit.

### Competing families
- **G0 grammar-only:** SELECT source-neutral or driven solely by observable context; no hidden prose.
- **GL structured source:** reusable latent source items with language-like frequency and local sequence dependence, optionally authentic Latin/Italian/German source controls; FORM/SELECT pipeline shared.
- **GN codebook/notation:** recurrent short motifs and constrained transitions without natural-language lexicon; same renderer and capacity.
- **GT table/repertoire:** heterogeneous section-conditioned source types with weak sequential dependence; deliberately hostile non-language control.

The word "language-like" here denotes a *simulation family*, not an ancient-language identification. Hiding source labels prevents selection on truth.

## Stage 0 (run now): exact renderer transport oracle ceiling
Question: **Even with the correct encoder and source transition matrix supplied, can a latent structured vocabulary (32 or 64 distinct source items) be recovered after collapsing through 8 low-rank SELECT controllers and the strictly frozen FORM renderer?**
- Use the archived frozen renderer code `select_form_socket_oracle_phaseK0_20261004.py`: K12 piece classes; strict K8 continuation groups; legal destination graph; exact-piece START/CONT distributions; depth- and incoming-class-conditioned STOP.
- Generate 3 source families (GL, GN, GT) and grammar-only negative (G0), with 32 and 64 source types; positive type recurrences and context; SELECT ENTRY strength 5, ROUTE 0.5, rank 2, fixed across a token.
- Source types share just 8 SELECT controller distributions, so instantaneous lexical identity is not automatically recoverable. Correct source transition graph and SELECT parameters provided only in stage 0 to establish an optimistic *upper bound*.
- Compare Viterbi/posterior source identity, NMI/ARI, posterior group accuracy, exact marginal sequential log evidence over IID-source baseline, same word tokens with order shuffled. Preserve the same distribution and FORM mechanism across families.
- A perfect decoder on stage 0 **does not qualify blind inversion**; even stage-0 failure precludes interpreting later real-text negatives. No source label or external dictionary is involved.

**Stage 0 gate:** On at least 20 fresh seeds per family at the target length, median oracle source NMI >=0.70, 10th percentile >=0.60, >=90% replicates NMI >=0.60. Also assess source-type pairwise recurrence precision/recall and source-family classification. First few seeds are *plumbing/pilot only*; do not relabel them confirmation.
If the 64-type model fails even with oracle parameters, retreat to coarser **identifiable equivalence classes** and test whether the information needed by GL/GN/GT can survive the renderer, rather than improving optimiser tricks.

## Next, only if stage 0 is powered
1. Stage 1: **blind** planted-source calibration. Use archived `SELECT_FORM_NORMALIZED_CALIBRATION_K4_20261004.md` envelope, no truth-selected hyperparameters, independent search/restart stability, ≥90% power and ≤5% FPR under structure-matched wrong-family/grammar-only nulls. Require false-family discrimination >=80%. Freeze decision thresholds before any real Voynich inspection.
2. Stage 2: factorized inverse estimate of SELECT-socket traces from running text (not raw glyph-to-letter substitution); evaluate structure-only vs GL/GN/GT on physical-bifolium heldout folds, treating line-entry, section, scribe and FORM legality as frozen nuisance. Preserve uncertainty in family equivalences.
3. Stage 3: positive formal residual tests. First quantify heldout novelty, source-type identity links and cross-word predictive bits *beyond* matched G0, not just dictionary hits. Then optionally score period-matched source-language corpora; language filtering remains secondary and cannot rank candidate grammars without independent controls.
4. Stage 4: actual semantics only after source mechanism, section-specific holdout, transcriptions and document-context anchors converge. Pharmaceutical labels, zodiac circular text and rosette/spatial inscriptions are different ecologies and **never** added to balneological running text for power.

## Avoid already known failure modes
- No normalization or splitting that leaks source identity from heldout observations or folio identity.
- Use physical bifolium folds and independent negative controls; do not train and evaluate on opposing pages of the same bifolium as if independent.
- Synthetic sources that are too easy or have visible artificial length codes do not qualify an instrument.
- K4 identifiability was problematic and cannot be declared solved by a higher likelihood or an oracle decode.
- No cross-ecology claims or direct labels for states prior to generative closure.
- If qualification fails, record a **failure of the instrument/family**, not "Voynich contains no language".

## Exit criteria
A table for every family: oracle ceiling; blind NMI/ARI; heldout predictive gain over G0; hard negative power/FPR; fold replication; legal FORM and stop consistency; what cannot be inferred. If no instrument meets gates, don't consume further GPU merely to expand the search.

Record seeds, corpus SHA, renderer commit, code, control manifests, hardware, and outcomes; never modify existing authoritative handoffs or main.
