# Voynich text structure: bifolia, ordering and word transformations

One place for the evidence behind our replies in voynich.ninja threads 6182 and 6199: the bifolium tests, what the
text can and cannot tell us about order, the Super Grammar results on word transformations, and the run records
they come from. Every number below is copied from a file in this folder or in `research/bifolium_order_tests/`.

**Read the status column.** Only items with a *certified* certificate are part of the frozen Super Grammar v03 kernel
([release note](../research/RELEASE_supergrammar_v03_20260920.md)). Everything marked *provisional* passed its own
held-out tests but has not been through the fail-closed certification and clean-room replay. *Rejected* means tested
and rejected. Nothing here is a decipherment or a claim about meaning.

## What is in the folder

| Path | Contents |
|---|---|
| [`../research/bifolium_order_tests/`](../research/bifolium_order_tests/) | Runnable code + inputs + expected outputs for the bifolium, ordering-calibration and Echo-chain tests (`./run_all.sh`, about an hour) |
| [`figures/`](figures/) | The five figures from the thread 6199 reply |
| [`supergrammar/certification/`](supergrammar/certification/README.md) | The v03 certificates: 9 certified theorems, the lemmas each depends on, 141 claim certificates (32 certified), 2,537 proof obligations, the evidence rows behind them, release checks, replay jobs and retractions |
| [`supergrammar/claims_register_v03.csv`](supergrammar/claims_register_v03.csv) | All 147 Super Grammar claims with statement, evidence, falsifier, provenance and `certificate_status` (also `.json`) |
| [`supergrammar/key_claims.md`](supergrammar/key_claims.md) | The claims that bear on the questions below, in full |
| [`supergrammar/theorems_v03.csv`](supergrammar/theorems_v03.csv) | Theorem wrappers and their status |
| `supergrammar/npcompl_regime_*.csv` | The test of the five-regime partition against Davis hands |
| [`supergrammar/lexeme_candidates_v01.csv`](supergrammar/lexeme_candidates_v01.csv) | 107 candidate word units with their support class and forward/reverse gains |
| [`records/`](records/INDEX.md) | 20 verbatim run records (pre-registrations and results), each with its retraction block at the top |

## The five open questions from thread 6199

### 1. A more comprehensive set of word transformations

- **Portable one-edit operator library (SG64, provisional).** An exhaustive one-edit graph over all 5,076 running-text
  EVA types. Operators were discovered on Herbal A vs Herbal B only, then replayed on Pharmaceutical-A vs
  Biological+Stars-B. 21 operator classes keep their A/B direction and clear 2 null SD in both. Strongest: insert *d*
  before final *y* (discovery 6.87 SD, validation 4.83 SD); pre-final *a/o*, internal *e/o*, *o* before *d*, pre-final *d/e*
  all 3.6 to 5.3 SD. Record: [`records/voynich_supergrammar_v04_realization_operator_library_20260924.md`](records/voynich_supergrammar_v04_realization_operator_library_20260924.md).
- **Currier-conditioned realisation (`SG47_CURRIER_CONDITIONED_REALIZATION_LAYER`, provisional).** 45 Xy/Xdy minimal-pair families shift A to B on held-out
  data (5.82 SD); the A/C/B interaction is 8.39 SD and holds in ZL, Takahashi and a third transliteration. The chain is
  stepwise through C: *ey→edy* 4.7% → 20.7% → 61.8%; *chy→chdy* 26.7% → 31.8% → 68.4%. Record:
  [`records/voynich_supergrammar_v04_currier_realization_20260924.md`](records/voynich_supergrammar_v04_currier_realization_20260924.md).
- **Echo rules tested from outside (5 Oct).** Echo's core rules form a real network of correspondences aligned with
  A/B, but the A→B *direction* does not resolve under the strict control (z 1.78 ZL, 1.91 Takahashi), and family-specific
  effects drop below 2 SD under the strictest morphology-matched control. Record:
  [`records/voynich_echo_collision_20261005.md`](records/voynich_echo_collision_20261005.md).
- **Transfer to unseen word groups (BR2).** Section-conditioned transformations learned in one neighbourhood do not
  carry to unseen ones (ZL −3.00 SD, Takahashi −1.55 SD). Record:
  [`records/voynich_BR2_section_transform_transfer_closeout_20261006.md`](records/voynich_BR2_section_transform_transfer_closeout_20261006.md).

### 2. Changes across clusters and bifolia

- **The five-regime partition (`SG47_NPCOMPL_MULTI_REGIME`, OPEN_HELD).** (Two different claims share the SG47 number.) Useful descriptively, not resolved as a held-out
  grammar layer: regime model vs parent z 1.33; regimes vs Davis hands z 0.38 (`supergrammar/npcompl_regime_contrasts.csv`).
- **Currier vs hand (SGT06, certified bounded).** Currier A/B cannot be identified independently of section and Davis
  hand; no section+hand stratum contains both A and B.
- **Page and leaf (SG08, provisional).** The subword regime is strongly local at page and especially leaf scale.
- **Bifolium.** As an independent *linguistic state*, rejected (SG09: appears in trigram counts, collapses on word sets
  and under a strict leaf-pair control). As a *repertoire* unit, provisional (SG57: reproducible in 8 of 8 herbal quires
  held out, exact p .0078). Our glyph-transition tests agree with the trigram-level reading: conjugate halves are closer
  (z 7.1; 6.4 with equal-length pages; shuffled control 0.7), and the effect follows the sheet, not the opening.
- **Linear drift or reuse?** Subword regimes recur on distant pages and quires after matching Currier, hand and
  illustration (SGK12, z 2.9 and 3.7 in the two split directions; excluded from certification as not yet replayed). That
  argues for reuse of earlier states alongside any drift.
- **Can the order be recovered from the text?** Calibrated on two texts of known order cut to VMS sheet sizes, the
  best method finds about 1 true neighbour in 5 (chance 4%). Inside Herbal A, scribe 1, the order built from one leaf
  of each sheet agrees with the order from the other leaf (z 3.5 incidence seriation, 2.7 spectral), and middle sheets
  are no more stable than end sheets (Spearman 0.10, p 0.68). On eight dated Middle Low German manuscripts the same
  kind of transition distance dates books to about 38 years (3.15 SD), but against a text-family-aware baseline the
  gain is 1.59 SD and fails the family-independent gate; the audit of eo777's eight-manuscript graphic gives 1.75 SD
  under a family-matched null. Records: [`records/voynich_ren_v11_primary_historical_heldout_chronology_20261011_v01.md`](records/voynich_ren_v11_primary_historical_heldout_chronology_20261011_v01.md),
  [`records/voynich_eo777_eight_manuscript_chronology_controls_20261011_v01.md`](records/voynich_eo777_eight_manuscript_chronology_controls_20261011_v01.md).
- **Echo chain stages vs mixing.** On the chain's own pages the A→B step coincides with a change of scribe. Stage 1.5a is
  not distinguishable from a mix of the pages before and after it on the full chain (z 0.28); within scribe 1 it is (z 3.0).
  See `research/bifolium_order_tests/scripts/s30_chain_mixture.py`.

### 3. How changes propagate between rare and common words

- Collapsing the top 8 operators reduces word types by 24.8% and hapax types by 23.1% (SG65), so the operators reach rare
  forms at about the same rate as forms overall.
- But the families help only common words. In held-out prediction, rare strata get worse under the collapse (unseen
  forms −0.185, training hapax −0.256, frequency 2–4 −0.176 bits/token) while common forms improve. Record:
  [`records/voynich_latent_transform_discriminator_20260928.md`](records/voynich_latent_transform_discriminator_20260928.md).
- A direct test of how a change spreads from common to rare forms along the chain has **not** been run.

### 4. Which words are affected by which changes

- The eight highest-coverage operators touch 65.9% of running-text tokens (74.1% with the gallows *k/t*, *p/f*
  families). The operator codes are listed in the operator-library record; high-confidence pairs include
  chey→chedy, shey→shedy, qokey→qokedy, sheey→sheedy, chy→chdy, qokeey→qokeedy, keey→keedy, okey→okedy.
- On the Echo chain, the forms concentrated at stage 2a are half Echo rule outputs (15 of 30; exploratory), consistent
  with Timm & Schinner's documented chol → cheol → cheo → chey → chedy path.

### 5. Do the transformations preserve information or meaning?

- **Same contexts?** Distinct word types do share external contexts (context-first test: one group {aiiin, aiin, ain,
  air, ar, or, r, s}; final 7.3 SD), and pooling them still helps after a topology-matched control (3.4 SD). Echo-linked
  forms specifically do not show context equivalence under the strict control (0.25 SD). Records:
  [`records/voynich_context_first_CF3_20261005.md`](records/voynich_context_first_CF3_20261005.md), [`CF4`](records/voynich_context_first_CF4_topology_20261005.md).
- **One hidden root?** Collapsing the operator families removes 29.5% of the Currier A/B vocabulary difference against
  17.2% for matched random collapses (14.2 SD; SG66, provisional), but the same collapse is no better than random on a
  fresh Stars early/late split (+0.99 SD) and makes positional anomalies worse, not better.
- **Removable encoding noise?** Pre-registered state-aware and hidden-state transducers fail to turn the operator
  families into a simpler latent text; what survives is a weak causal state (about 2 SD) that behaves like part of how
  the surface is produced, not a wrapper around plaintext. Records:
  [`state-aware`](records/voynich_state_aware_transducer_result_20260928.md),
  [`causal hidden-state`](records/voynich_causal_hidden_state_transducer_result_20260928.md),
  [`ciphertext transducer`](records/voynich_hidden_state_transducer_result_20260928.md).
- **Rejected:** a reusable paragraph-level content state (SG25) and a stable code unit that transfers across scribes (SG33).

So far: the transformations are real formal relations between spellings; whether they keep meaning is not shown.

## Reproducing

- Bifolium, ordering and chain tests: `cd research/bifolium_order_tests && ./run_all.sh` (Python 3.10+, numpy,
  scipy, pandas, matplotlib).
- Super Grammar v03 kernel certifiers: `research/supergrammar_v03_*.py`.
- The run records name their own scripts and commits; most are in `research/`. Some runs used Hugging Face jobs and a
  Supabase database whose tables are referenced by name; the claims and test tables those runs wrote are exported here.

## Provenance and caveats

- Claims and run records were produced across Claude and ChatGPT sessions under a pre-registration protocol with
  matched nulls; the provenance field of each claim says where it came from. Records are exported as written,
  including their own retractions.
- Transliterations differ between runs (ZL, Takahashi, others); each record states which it used.
- Claims, theorems and records exported from the research database on 14 October 2026; certification tables on 15 October 2026.
