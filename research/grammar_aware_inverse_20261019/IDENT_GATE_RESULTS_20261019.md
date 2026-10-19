# IDENT-GATE-1 outcomes and decision (2026-10-19)

**Scientific status: the equivalence gate passed by construction; the proposed anchor detector FAILED its registered power and FPR gates. Not a Voynich decipherment and not a claim of no underlying language.** This programme is a methodological correction to Q1B; it does **not** supersede the Q1B numerical record.

## Why this instrument, not another pair-AUC decoder?
Prior Q1B registered 20-seed blind evaluations established an approximately .845–.870 AUC for shared **designed SELECT controller** classes, with ≈.498 source-neutral null, while source-identity *within each identical emission class* was ≈.50. Those facts are compatible both with language/source content and with copyist machinery or a constrained word-generation protocol. A new classifier of category co-membership cannot decide the cause. Even infinitely precise inference of an arbitrary latent class need not identify semantics.

The underlying causal symmetry is:
```
Semantic latent Z -> SELECT control C -> FORM/CONNECT/LINE_ENTRY -> observable Y
Production latent Z' -> SELECT control C -> FORM/CONNECT/LINE_ENTRY -> observable Y
```
If `Z` and `Z'` have identical stochastic laws and apply identical output operations, `P(Y | Semantic model)=P(Y | Production model)` for **all** Y. Relabelling the same latent mechanism does not create a testable difference. No test restricted to Y can have power greater than its false-positive rate for this pair. This is a mathematical limitation of the defined alternatives, not a substantive hypothesis about which generated the historical VMS.

## Implemented and tested
[Preregistration](IDENT_GATE_PREREG_20261019.md); [code](ident_gate.py) frozen at commit `bfc94481e79c530de644d96a62c5e298ff60f586`, reuses FORM source pinned at `fa4321e75365d83c7ea9d07b0d6c24ee77c01b2f` (57 exact pieces; K12 start/destination class, 8 response groups; END/CONTINUE explicitly scored). Synthetic only, 5 packet folds, 32 lines/fold, 10 forms/line, 20 fresh seeds 20271100–20271119. Withheld fold 4 never used for model selection.

- Paired source/production interpretations have **exactly identical synthetic surface SHA256 hashes** and marginal FORM/latent-state log likelihoods for 20/20 seeds, maximum |difference| = **0**. Surface is not merely statistically similar: the two named models are the same observable process. This is an intentional rigorous adversarial counterexample.
- Added synthetic "independent anchor" observation per line. Arm A1 links anchor to latent state (called "semantic"); A2 links it to the corresponding copyist/production state. These have **identical joint surface–anchor observations** by deterministic coupling; no classifier can tell which interpretation produced the data.
- Arm A0 draws an anchor independently of source after section/entry stratification. Frozen baseline predicts A from nuisance section/entry alone; candidate adds aggregate START and ROUTE FORM-piece observables. Both use same logistic family, validation-selected C, and heldout packet fold.
- **Results**, [all 20 cases and code/version hashes](results/ident_gate_20seeds_20261019.json):

| Synthetic anchor arm | Median heldout improvement bits/line | Median heldout AUC | Runs with gain > +0.02 |
|---|---:|---:|---:|
| A1 synthetic semantic | **+0.008006** | **0.5720** | 6/20 |
| A2 matched copyist/production HARD negative | **+0.008006** | **0.5720** | 6/20 |
| A0 stratified independent | **−0.006701** | **0.5484** | 2/20 |

Gate declared pre-run: ≥18/20 A1 detected and ≤1/20 A0 false positive. **FAIL BOTH:** positive 6/20; independent control 2/20. A2 is **indistinguishable by construction**, successfully exposing the fundamental source-vs-production identifiability limitation. The AUCs are low and not reliable semantic signals. Do not tune these data post hoc or call this a qualification. The exact twin equivalence is proved even if anchor-model classification remains underpowered.

One 12-lines/fold, 1-seed smoke run preceded the fresh 20-seed panel and was not counted.

## Existing independent-anchor inventory — READ-ONLY SQL audit
Source: Supabase `research-agent` project `ymaqlcfjmdwncdbjprmw`, tables `vms_catalogue.vms_objects`, `vms_catalogue.vms_folios`, `public.vms_bio_masks`, `public.vms_manual_masks`, `public.cmp_balneis_sections`, `public.cmp_balneis_concordance`. These are live counts as queried 2026-10-19; re-run before any empirical study. No DB writes.

- Balneological: **44 human_figure objects over 3 folios**, none `reviewed=true`. Independently, **92 public.vms_bio_masks** across 4 folios, geometric segmentation, not validated semantics.
- Herbal-A: **272 objects / 105 folios**, only **25 reviewed** (17 root, 8 plant). Those categories say object part, **not identifiable plant species nor Latin gloss**. Herbal-B 19 objects across 10 folios, none reviewed.
- Stars: 38 star objects across 3 folios, none reviewed; masking is not language annotation.
- Manual masks: **416 across 150 folios**, image-component segmentation; do not mistake high coverage for semantic ground truth.
- External De Balneis: **37** editorially described sections, **407** witness-section concordance entries. These anchor historical prose and illustration to each other *inside De Balneis*, not to specific Voynich words. They are useful candidates for an independent witness-matching experiment, not gold translation pairs.
- Conclusion: the available catalogue does not contain sufficiently independently validated, folio-localised **text-to-meaning** matches to qualify a semantic inverse solver. Building a supervised VMS decoder here would train on unverified guesses.

Queries archived [anchor_audit_readonly.sql](anchor_audit_readonly.sql).

## Next architecture decision
**Do not build another unsupervised decoder or increase latent state count.** Build an **anchor admissibility registry** before a semantic inversion: every candidate anchor must have (1) independent period witness/reviewer-proven label, (2) demonstrable spatial alignment to an independently segmented running-text unit rather than general folio resemblance, (3) a narrow specified lexical/semantic prediction that survives FORM and CONNECT, (4) a same-section/hand/folio-position matched negative, and (5) an independently held-out physical fold and exact time order. Do not mix Pharma label text with Q13 running.

Initial realistic target may be a *single narrow witness*: one proposed plant/root identity that appears in multiple **independently documented** illustrations and has a stable adjacent label, or a De Balneis scene with a specific attested feature and Voynich localization beyond layout. Evaluate even one such alignment only after an independent iconographic reviewer confirms both sides.

The central distinction is between **detecting remaining predictive information** (possible from text alone) and **attributing that information to semantic encoding** (requires evidence that discriminates language/content from production). Earlier RESIDREC programmes already pursued the former. Do not rerun them under a new name, and do not infer the latter from the former.

Disposition: FORM source/production interpretive mechanism **NON-IDENTIFIABLE under paired models**; synthetic anchor classifier **INSTRUMENT FAILED**; real-VMS semantic inverse **NOT LICENSED**. No GPU running. Existing authoritative handoffs left untouched.
