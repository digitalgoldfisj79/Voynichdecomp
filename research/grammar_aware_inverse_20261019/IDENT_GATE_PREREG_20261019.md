# IDENT-GATE-1 — causal identifiability before another Voynich inverse solver
Date: 2026-10-19. Design frozen before new simulations.
Status: an **instrument audit**, not decipherment, and not permission to infer source semantics.

## Reconsideration of the proposed relationship solver
Q1B already showed pairwise same-SELECT-class AUC 0.845–0.870 on synthetic source-bearing sequences, with source-independent null ≈0.498; within-controller source identity AUC ≈0.5; all 8-class NMI gates failed. Those results are **expected given the constructed forward channel**. Another classifier trained to recognise shared emission behaviour would largely retest that same tautology.

More fundamentally, the causal labels "hidden natural-language source" and "hidden copying/production regime" can induce exactly the same marginal distribution over Voynich tokens. If both implement the identical stochastic SELECT controller, FORM, CONNECT and LINE_ENTRY mappings, they are observationally equivalent even with infinite data or infinite compute. A full-likelihood classifier cannot tell them apart. Reparametrisation and label permutation are not discoveries.

## Gate A: exact observational equivalence
- Model L: a persistent line-level latent semantic category drives SELECT; FORM renders exact 57-piece/12-class/8-response grammar with strict continuation and STOP from pinned source `fa4321e75365d83c7ea9d07b0d6c24ee77c01b2f`.
- Model P: **the same stochastic latent process** is interpreted as a persistent production/copying regime, not encoded meaning. All emissions and nuisance variables are mathematically identical.
- Use coupled fixed seeds and verify exact full surface trace identity (SHA256). Compare exact forward likelihoods under both mechanisms. Required difference <1e-10 and equality at all observations. Verdict: NOT IDENTIFIABLE (by construction) even if language-like sequential structure exists.
- There is no valid claim about whether real Voynich is L or P after this result.

## Gate B: independent-anchor necessity and insufficiency
Build *synthetic* folios (5 non-overlapping packet folds, physical-line units, each with 8–12 tokens), with several fixed section and line-entry strata. Per-line latent state determines exact FORM emissions through the pinned renderer. Diagnostic anchor A (e.g. abstract diagram class) is:
  - A0: independent anchor after stratifying section / line and fold => known negative.
  - A1: linked to a putative semantic line concept => synthetic positive.
  - A2: linked to a production/copied-session latent state => HARD NEGATIVE with the **same joint distribution of (surface, anchor)** as A1.
For A1 and A2 use the same coupled random innovations: they are not distinguishable from data because the anchor correlation itself could be downstream of common production provenance.
- Baseline: anchor from section/line-entry metadata only (fit TRAIN).
- Full: anchor from exactly the same metadata + FORM surface summary (fit TRAIN). Blind validation hyperparameter selection, outer fold held out by packet, no fold IDs in features; report paired TEST log-loss difference (bits/line), anchored classifier AUC.
- Sensitivity expected A1>A0, but **A2=A1** exactly (no historical promotion). Calibrate over 20 new seeds, no cherry picking.

## Gates and decision
1. Surface-equivalence SHA and model logscore equality. If fail, debug construction; do not claim mechanisms differ.
2. Anchor instrument sensitivity >=90% of 20 A1 runs exceed a frozen +0.02 bits/line of heldout gain, AND <=5% of A0 runs exceed +0.02. A2 must score indistinguishably to A1 under exact coupling; otherwise bug or leakage.
3. This is **not** a controlled test of real CONNECT/LINE_ENTRY: only nuisance-inspired contextual proxy features and the strict original FORM are present. The actual fitted contextual constraints would be required for a real renderer-native test.
4. The credible next stage must therefore use genuinely **independent discriminating anchors** (period-specific plaintext alignments, stable iconographic labels that cannot be explained by layout/bifolium production, or a historically grounded codebook). Merely adding visually correlated labels does not suffice, as A2 demonstrates.
5. Hard-negative matching must preserve section, scribe, physical packet, line-entry, word lengths, positional FORM legality, and CONNECT dependencies. No cross-type mixing (Pharma labels / circular / running).
6. No real Voynich decoding or claims until a positive and a hard negative are *observably separable*, the proposed instrument passes power/FPR, and heldout physical packets replicate. If mechanisms remain observationally identical, record NON-IDENTIFIABLE rather than chasing optimisers.

Artifacts: source script, hashes, per-seed outcomes, failure report, project GitHub branch only. Existing handoffs untouched.
